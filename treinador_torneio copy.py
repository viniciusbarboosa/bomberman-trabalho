import sys
import random
import importlib
import signal
import time
import torch

# ── Configurações (Mantidas do seu original) ──────────────────────────────────
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
print("=" * 55)
print("  BOMBERMAN RL — TREINADOR NEURAL")
print("=" * 55)

nomes  = ["ia_jogador1", "ia_jogador2", "ia_jogador3", "ia_jogador4"]
ia_fns = []
for nome in nomes:
    try:
        mod = importlib.import_module(nome)
        ia_fns.append(mod.decidir_acao)
        print(f"  ✓ {nome}")
    except Exception as e:
        print(f"  ✗ {nome} — {e}")
        ia_fns.append(None)

ia1_mod = importlib.import_module("ia_jogador1")

# ══════════════════════════════════════════════════════════════════════════════
#  CLASSES DE SIMULAÇÃO (Bomba e Player adaptados para treino)
# ══════════════════════════════════════════════════════════════════════════════

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
            if self.tempo_explosao <= 0: self._explodir(mapa, bombas)
        else: self.tempo_fogo -= delta

    def _explodir(self, mapa, bombas):
        self.explodida = True
        self.tempo_fogo = TEMPO_FOGO
        self.fogo.append((self.x, self.y))
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            for i in range(1 + (self.nivel - 1) * 2):
                nx, ny = self.x + dx*(i+1), self.y + dy*(i+1)
                if not (0<=nx<COLS and 0<=ny<ROWS) or mapa[ny][nx] == 2: break
                self.fogo.append((nx, ny))
                if mapa[ny][nx] == 1: break
        for fx, fy in self.fogo:
            for b in bombas:
                if not b.explodida and (b.x, b.y) == (fx, fy): b._explodir(mapa, bombas)

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

    def get_self_state(self):
        return {'grid_x': self.grid_x, 'grid_y': self.grid_y, 'max_bombas': self.max_bombas,
                'bombas_ativas': len(self.bombas), 'bomba_nivel': self.bomba_nivel, 'ativo': self.ativo}

    def decidir(self, mapa, players, bombas, tempo, pontos):
        if not self.ativo or self.ia_fn is None: return "parado"
        return self.ia_fn(self, mapa, players, bombas, tempo, pontos, HUD_INFO, self.get_self_state())

    def mover(self, dx, dy):
        self.dest_x, self.dest_y = self.grid_x + dx, self.grid_y + dy
        self.movendo = True
        self.tempo_mov = 0.0

# ══════════════════════════════════════════════════════════════════════════════
#  SISTEMA DE PARTIDA
# ══════════════════════════════════════════════════════════════════════════════

def simular_partida():
    mapa = [linha[:] for linha in MAPA_BASE]
    posicoes = [(0,0),(12,0),(0,10),(12,10)]
    players = [Player(px, py, ia_fns[i], i) for i, (px, py) in enumerate(posicoes)]
    bombas, pontos, tempo = [], [0]*4, TEMPO_PARTIDA

    while tempo > 0 and sum(p.ativo for p in players) > 1:
        tempo -= DELTA_FIXO
        for p in players:
            if not p.ativo: continue
            if p.movendo:
                p.tempo_mov += DELTA_FIXO
                if p.tempo_mov >= 0.1:
                    p.grid_x, p.grid_y, p.movendo = p.dest_x, p.dest_y, False
                continue

            acao = p.decidir(mapa, players, bombas, tempo, pontos)
            dx, dy = 0, 0
            if acao == "cima": dy = -1
            elif acao == "baixo": dy = 1
            elif acao == "esquerda": dx = -1
            elif acao == "direita": dx = 1
            elif acao == "bomba" and len(p.bombas) < p.max_bombas:
                if not any(b.x==p.grid_x and b.y==p.grid_y and not b.explodida for b in bombas):
                    nova_b = Bomba(p.grid_x, p.grid_y, p.bomba_nivel, p)
                    bombas.append(nova_b); p.bombas.append(nova_b)

            if (dx or dy):
                nx, ny = p.grid_x+dx, p.grid_y+dy
                if 0<=nx<COLS and 0<=ny<ROWS and mapa[ny][nx] in [0,3,4]:
                    if not any(b.x==nx and b.y==ny and not b.explodida for b in bombas):
                        p.mover(dx, dy)

        # Atualizar bombas e fogos
        for b in bombas[:]:
            b.atualizar(DELTA_FIXO, mapa, bombas)
            if b.explodida and b.tempo_fogo <= 0:
                for fx, fy in b.fogo:
                    if mapa[fy][fx] == 1:
                        mapa[fy][fx] = (3 if random.random() < 0.1 else 0)
                        if b.dono: pontos[b.dono.idx] += PONTOS_BLOCO
                bombas.remove(b)
                if b.dono and b in b.dono.bombas: b.dono.bombas.remove(b)

        # Morte por fogo e Powerups
        for b in bombas:
            if b.explodida:
                for p in players:
                    if p.ativo and (p.grid_x, p.grid_y) in b.fogo:
                        p.ativo = False
                        if b.dono and b.dono != p: pontos[b.dono.idx] += PONTOS_MATAR_JOGADOR

        for p in players:
            if p.ativo and not p.movendo:
                if mapa[p.grid_y][p.grid_x] == 3:
                    p.max_bombas += 1; mapa[p.grid_y][p.grid_x] = 0; pontos[p.idx] += 200

    vencedores = [p for p in players if p.ativo]
    idx_vencedor = vencedores[0].idx if len(vencedores) == 1 else pontos.index(max(pontos))
    return idx_vencedor, pontos

# ══════════════════════════════════════════════════════════════════════════════
#  LOOP DE TREINO
# ══════════════════════════════════════════════════════════════════════════════

parar = False
def handler_ctrl_c(sig, frame):
    global parar
    parar = True
signal.signal(signal.SIGINT, handler_ctrl_c)

vitorias = [0]*4
partida = 0
print("\nIniciando Treino Neural... Pressione Ctrl+C para encerrar e salvar.")

while not parar:
    venc, pts = simular_partida()
    vitorias[venc] += 1
    partida += 1
    
    if partida % 10 == 0:
        print(f"Partida: {partida} | IA1 Vitórias: {vitorias[0]} ({vitorias[0]/partida*100:.1f}%) | Placar: {vitorias}")

print("\nFinalizando e Salvando Pesos...")
# Chama o salvamento da rede neural (que definimos no ia_jogador1)
torch.save(ia1_mod._brain.modelo.state_dict(), "ia_neural.pth")
print("Cérebro salvo em ia_neural.pth")