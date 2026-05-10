"""
treino.py — Treinador Neural para Bomberman (DQN melhorado)
- Double DQN + Replay Buffer + Target Network
- Epsilon decay por episódio
- Salvamento automático periódico
- Estatísticas detalhadas
"""

import sys
import random
import importlib
import signal
import time
import torch

# ── Configurações ──────────────────────────────────────────────────────────────
TEMPO_PARTIDA  = 180
DELTA_FIXO     = 0.1
TEMPO_EXPLOSAO = 4.0
TEMPO_FOGO     = 0.5
ROWS, COLS     = 11, 13
MAX_BOMBAS     = 5

PONTOS_BLOCO             = 100
PONTOS_POWERUP_COLETADO  = 200
PONTOS_POWERUP_DESTRUIDO = -50
PONTOS_MATAR_JOGADOR     = 1000
PONTOS_VITORIA           = 10000
PROB_BOMBA               = 0.12
PROB_FOGO                = 0.10

SALVAR_A_CADA  = 50     # partidas entre cada salvamento automático
LOG_A_CADA     = 10     # partidas entre cada linha de log

MAPA_BASE = [
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    [0, 2, 1, 2, 0, 2, 0, 2, 1, 2, 1, 2, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
    [1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 0, 2, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
]

HUD_INFO = {
    'tile_size': 48, 'rows': ROWS, 'cols': COLS,
    'tempo_movimento': 0.1, 'tempo_explosao': TEMPO_EXPLOSAO,
    'tempo_fogo': TEMPO_FOGO, 'max_bombas': MAX_BOMBAS,
}

# ── Carrega IAs ────────────────────────────────────────────────────────────────

print("=" * 60)
print("  BOMBERMAN — TREINADOR NEURAL (DQN MELHORADO)")
print("=" * 60)

nomes  = ["ia_jogador1", "ia_jogador2", "ia_jogador3", "ia_jogador4"]
ia_mods = []
ia_fns  = []

for nome in nomes:
    try:
        mod = importlib.import_module(nome)
        ia_mods.append(mod)
        ia_fns.append(mod.decidir_acao)
        print(f"  ✓ {nome}")
    except Exception as e:
        print(f"  ✗ {nome} — {e}")
        ia_mods.append(None)
        ia_fns.append(None)

ia1_mod = ia_mods[0]
print()

# ── Classes de Simulação ───────────────────────────────────────────────────────

class Bomba:
    def __init__(self, x, y, nivel=1, dono=None):
        self.x, self.y, self.nivel, self.dono = x, y, nivel, dono
        self.tempo_explosao = TEMPO_EXPLOSAO
        self.explodida = False
        self.tempo_fogo = 0.0
        self.fogo = []

    def atualizar(self, delta, mapa, bombas):
        if not self.explodida:
            self.tempo_explosao -= delta
            if self.tempo_explosao <= 0:
                self._explodir(mapa, bombas)
        else:
            self.tempo_fogo -= delta

    def _explodir(self, mapa, bombas):
        self.explodida = True
        self.tempo_fogo = TEMPO_FOGO
        self.fogo.append((self.x, self.y))
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            for i in range(1 + (self.nivel - 1) * 2):
                nx, ny = self.x + dx*(i+1), self.y + dy*(i+1)
                if not (0 <= nx < COLS and 0 <= ny < ROWS) or mapa[ny][nx] == 2:
                    break
                self.fogo.append((nx, ny))
                if mapa[ny][nx] == 1:
                    break
        for fx, fy in self.fogo:
            for b in bombas:
                if not b.explodida and (b.x, b.y) == (fx, fy):
                    b._explodir(mapa, bombas)


class Player:
    def __init__(self, x, y, ia_fn, idx):
        self.grid_x, self.grid_y = x, y
        self.ia_fn, self.idx = ia_fn, idx
        self.ativo = True
        self.max_bombas = 1
        self.bombas = []
        self.bomba_nivel = 1
        self.movendo = False
        self.tempo_mov = 0.0
        self.dest_x, self.dest_y = x, y
        # Para o time (necessário para o checksum do jogo real)
        self.time = idx % 2

    def get_self_state(self):
        return {
            'grid_x': self.grid_x, 'grid_y': self.grid_y,
            'max_bombas': self.max_bombas, 'bombas_ativas': len(self.bombas),
            'bomba_nivel': self.bomba_nivel, 'ativo': self.ativo,
        }

    def decidir(self, mapa, players, bombas, tempo, pontos):
        if not self.ativo or self.ia_fn is None:
            return "parado"
        return self.ia_fn(self, mapa, players, bombas, tempo, pontos, HUD_INFO, self.get_self_state())

    def mover(self, dx, dy):
        self.dest_x, self.dest_y = self.grid_x + dx, self.grid_y + dy
        self.movendo = True
        self.tempo_mov = 0.0


# ── Simulação de Partida ───────────────────────────────────────────────────────

def simular_partida():
    mapa    = [linha[:] for linha in MAPA_BASE]
    posicoes = [(0,0),(12,0),(0,10),(12,10)]
    players  = [Player(px, py, ia_fns[i], i) for i, (px, py) in enumerate(posicoes)]
    bombas   = []
    pontos   = [0] * 4
    tempo    = TEMPO_PARTIDA

    while tempo > 0 and sum(p.ativo for p in players) > 1:
        tempo -= DELTA_FIXO

        for p in players:
            if not p.ativo:
                continue

            # Resolve movimento em andamento
            if p.movendo:
                p.tempo_mov += DELTA_FIXO
                if p.tempo_mov >= 0.1:
                    p.grid_x, p.grid_y = p.dest_x, p.dest_y
                    p.movendo = False
                    # Coleta powerup
                    if mapa[p.grid_y][p.grid_x] == 3:
                        p.max_bombas = min(p.max_bombas + 1, MAX_BOMBAS)
                        mapa[p.grid_y][p.grid_x] = 0
                        pontos[p.idx] += PONTOS_POWERUP_COLETADO
                    elif mapa[p.grid_y][p.grid_x] == 4:
                        p.bomba_nivel = min(p.bomba_nivel + 1, 4)
                        mapa[p.grid_y][p.grid_x] = 0
                        pontos[p.idx] += PONTOS_POWERUP_COLETADO
                continue

            # Decisão da IA
            acao = p.decidir(mapa, players, bombas, tempo, pontos)

            dx, dy = 0, 0
            if acao == "cima":     dy = -1
            elif acao == "baixo":  dy =  1
            elif acao == "esquerda": dx = -1
            elif acao == "direita":  dx =  1
            elif acao == "bomba":
                if len(p.bombas) < p.max_bombas:
                    if not any(b.x == p.grid_x and b.y == p.grid_y and not b.explodida for b in bombas):
                        nova = Bomba(p.grid_x, p.grid_y, p.bomba_nivel, p)
                        bombas.append(nova)
                        p.bombas.append(nova)

            if dx or dy:
                nx, ny = p.grid_x + dx, p.grid_y + dy
                if 0 <= nx < COLS and 0 <= ny < ROWS and mapa[ny][nx] in [0, 3, 4]:
                    if not any(b.x == nx and b.y == ny and not b.explodida for b in bombas):
                        p.mover(dx, dy)

        # Atualiza bombas
        for b in bombas[:]:
            b.atualizar(DELTA_FIXO, mapa, bombas)
            if b.explodida and b.tempo_fogo <= 0:
                for fx, fy in b.fogo:
                    if mapa[fy][fx] == 1:
                        r = random.random()
                        mapa[fy][fx] = 3 if r < PROB_BOMBA else (4 if r < PROB_BOMBA + PROB_FOGO else 0)
                        if b.dono:
                            pontos[b.dono.idx] += PONTOS_BLOCO
                    elif mapa[fy][fx] in [3, 4]:
                        mapa[fy][fx] = 0
                        if b.dono:
                            pontos[b.dono.idx] += PONTOS_POWERUP_DESTRUIDO
                bombas.remove(b)
                if b.dono and b in b.dono.bombas:
                    b.dono.bombas.remove(b)

        # Mortes por fogo
        for b in bombas:
            if b.explodida and b.tempo_fogo > 0:
                for p in players:
                    if p.ativo and (p.grid_x, p.grid_y) in b.fogo:
                        p.ativo = False
                        if b.dono and b.dono != p:
                            pontos[b.dono.idx] += PONTOS_MATAR_JOGADOR
                        elif b.dono == p:
                            pontos[p.idx] = max(0, pontos[p.idx] - PONTOS_MATAR_JOGADOR)

    # Determina vencedor
    vivos = [p for p in players if p.ativo]
    if len(vivos) == 1:
        idx_v = vivos[0].idx
        pontos[idx_v] += PONTOS_VITORIA
    else:
        idx_v = pontos.index(max(pontos))

    return idx_v, pontos


# ── Loop de Treino ─────────────────────────────────────────────────────────────

parar = False

def handler_ctrl_c(sig, frame):
    global parar
    print("\n[!] Encerrando após a partida atual...")
    parar = True

signal.signal(signal.SIGINT, handler_ctrl_c)

vitorias    = [0] * 4
pontos_hist = [[] for _ in range(4)]   # histórico de pontos por IA
partida     = 0
tempo_inicio = time.time()

print("Iniciando treino... Pressione Ctrl+C para encerrar e salvar.\n")
print(f"{'Partida':>8} | {'IA1 Vit%':>8} | {'IA2 Vit%':>8} | {'IA3 Vit%':>8} | {'IA4 Vit%':>8} | {'ε':>6} | {'Buffer':>7} | {'Tempo/p':>7}")
print("-" * 90)

while not parar:
    venc, pts = simular_partida()
    vitorias[venc] += 1
    partida += 1

    for i in range(4):
        pontos_hist[i].append(pts[i])
        if len(pontos_hist[i]) > 100:
            pontos_hist[i].pop(0)

    # Decay do epsilon ao fim de cada partida
    if ia1_mod and hasattr(ia1_mod, "decay_epsilon"):
        ia1_mod.decay_epsilon()

    # Log periódico
    if partida % LOG_A_CADA == 0:
        eps  = ia1_mod._brain.epsilon if ia1_mod else 0
        buf  = len(ia1_mod._brain.buffer) if ia1_mod else 0
        t_p  = (time.time() - tempo_inicio) / partida

        pcts = [f"{vitorias[i]/partida*100:>7.1f}%" for i in range(4)]
        print(f"{partida:>8} | {pcts[0]:>8} | {pcts[1]:>8} | {pcts[2]:>8} | {pcts[3]:>8} | {eps:>6.3f} | {buf:>7} | {t_p:>6.2f}s")

    # Salvamento periódico
    if partida % SALVAR_A_CADA == 0:
        if ia1_mod and hasattr(ia1_mod, "salvar"):
            ia1_mod.salvar()

print("\n" + "=" * 60)
print("  ENCERRANDO TREINO")
print("=" * 60)
print(f"  Partidas jogadas : {partida}")
for i in range(4):
    pct = vitorias[i] / max(partida, 1) * 100
    med = sum(pontos_hist[i]) / max(len(pontos_hist[i]), 1)
    print(f"  IA{i+1}: {vitorias[i]} vitórias ({pct:.1f}%) | média pts (últ.100): {med:.0f}")

# Salvamento final
if ia1_mod and hasattr(ia1_mod, "salvar"):
    ia1_mod.salvar()

print("\nConcluído.")