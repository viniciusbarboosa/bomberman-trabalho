import sys
import random
import importlib
import signal
import time

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
print("  BOMBERMAN — TREINO EM LOOP  controlc para")
print("=" * 55)
print("\nCarregando IAs...")

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

if ia_fns[0] is None:
    print("\nERRO: ia_jogador1 não encontrada")
    sys.exit(1)

ia1_mod = importlib.import_module("ia_jogador1")


# ══════════════════════════════════════════════════════════════════════════════
#  OBJETOS SIMULADOS
# ══════════════════════════════════════════════════════════════════════════════

class Bomba:
    def __init__(self, x, y, nivel=1, dono=None):
        self.x = x
        self.y = y
        self.nivel = nivel
        self.dono = dono
        self.tempo_explosao = TEMPO_EXPLOSAO
        self.explodida = False
        self.tempo_fogo = 0.0
        self.fogo = []
        self.anim_frame = 0
        self.anim_timer = 0

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
                nx = self.x + dx * (i + 1)
                ny = self.y + dy * (i + 1)
                if not (0 <= nx < COLS and 0 <= ny < ROWS): break
                if mapa[ny][nx] == 2: break
                self.fogo.append((nx, ny))
                if mapa[ny][nx] == 1: break
        for fx, fy in self.fogo:
            for b in bombas:
                if not b.explodida and (b.x, b.y) == (fx, fy):
                    b._explodir(mapa, bombas)


class Player:
    def __init__(self, x, y, time, ia_fn, idx):
        self.grid_x = x
        self.grid_y = y
        self.time = time
        self.ia_fn = ia_fn
        self.idx = idx
        self.ativo = True
        self.max_bombas = 1
        self.bombas = []
        self.bomba_nivel = 1
        self.movendo = False
        self.tempo_mov = 0.0
        self.dest_x = x
        self.dest_y = y
        self.pixel_x = x * 48
        self.pixel_y = y * 48
        self.ultima_direcao = "baixo"
        self.anim_frame = 0
        self.anim_timer = 0
        self.tipo = "ia"

    def get_self_state(self):
        return {
            'grid_x': self.grid_x, 'grid_y': self.grid_y,
            'max_bombas': self.max_bombas,
            'bombas_ativas': len(self.bombas),
            'bomba_nivel': self.bomba_nivel,
            'ativo': self.ativo,
        }

    def decidir(self, mapa, players, bombas, tempo, pontos):
        if self.ia_fn is None or not self.ativo:
            return "parado"
        try:
            return self.ia_fn(self, mapa, players, bombas, tempo,
                              pontos, HUD_INFO, self.get_self_state())
        except Exception:
            return "parado"

    def atualizar(self, delta):
        if self.movendo:
            self.tempo_mov += delta
            if self.tempo_mov >= 0.1:
                self.grid_x = self.dest_x
                self.grid_y = self.dest_y
                self.movendo = False
                self.tempo_mov = 0.0

    def mover(self, dx, dy):
        self.dest_x = self.grid_x + dx
        self.dest_y = self.grid_y + dy
        self.movendo = True
        self.tempo_mov = 0.0
        if dy==-1: self.ultima_direcao = "cima"
        elif dy==1: self.ultima_direcao = "baixo"
        elif dx==-1: self.ultima_direcao = "esquerda"
        elif dx==1: self.ultima_direcao = "direita"


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULAÇÃO DE UMA PARTIDA
# ══════════════════════════════════════════════════════════════════════════════

def criar_powerup():
    r = random.random()
    if r < PROB_BOMBA: return 3
    elif r < PROB_BOMBA + PROB_FOGO: return 4
    return 0


def simular_partida():
    mapa = [linha[:] for linha in MAPA_BASE]
    posicoes = [(0,0),(12,0),(0,10),(12,10)]
    players = [
        Player(px, py, i % 2, ia_fns[i] if i < len(ia_fns) else None, i)
        for i, (px, py) in enumerate(posicoes)
    ]
    bombas  = []
    pontos  = [0, 0, 0, 0]
    tempo   = TEMPO_PARTIDA

    while tempo > 0:
        tempo -= DELTA_FIXO

        for p in players:
            if not p.ativo: continue
            p.atualizar(DELTA_FIXO)
            if p.movendo: continue

            acao = p.decidir(mapa, players, bombas, tempo, pontos)
            dx = dy = 0
            if   acao == "cima":     dy = -1
            elif acao == "baixo":    dy =  1
            elif acao == "esquerda": dx = -1
            elif acao == "direita":  dx =  1
            elif acao == "bomba":
                if len(p.bombas) < p.max_bombas:
                    if not any(b.x==p.grid_x and b.y==p.grid_y and not b.explodida for b in bombas):
                        b = Bomba(p.grid_x, p.grid_y, p.bomba_nivel, p)
                        bombas.append(b)
                        p.bombas.append(b)

            if dx or dy:
                nx, ny = p.grid_x+dx, p.grid_y+dy
                if (0<=nx<COLS and 0<=ny<ROWS and mapa[ny][nx] in [0,3,4]
                        and not any(b.x==nx and b.y==ny and not b.explodida for b in bombas)):
                    p.mover(dx, dy)

        for b in bombas[:]:
            b.atualizar(DELTA_FIXO, mapa, bombas)
            if b.explodida and b.tempo_fogo <= 0:
                for fx, fy in b.fogo:
                    if mapa[fy][fx] == 1:
                        mapa[fy][fx] = criar_powerup()
                        if b.dono: pontos[b.dono.idx] += PONTOS_BLOCO
                    elif mapa[fy][fx] in [3,4]:
                        mapa[fy][fx] = 0
                        if b.dono: pontos[b.dono.idx] += PONTOS_POWERUP_DESTRUIDO
                bombas.remove(b)
                if b.dono and b in b.dono.bombas:
                    b.dono.bombas.remove(b)

        for b in bombas:
            if b.explodida and b.tempo_fogo > 0:
                for p in players:
                    if p.ativo and (p.grid_x, p.grid_y) in b.fogo:
                        p.ativo = False
                        if b.dono == p:
                            pontos[p.idx] = max(0, pontos[p.idx] - PONTOS_MATAR_JOGADOR)
                        elif b.dono:
                            pontos[b.dono.idx] += PONTOS_MATAR_JOGADOR

        for p in players:
            if not p.ativo or p.movendo: continue
            tile = mapa[p.grid_y][p.grid_x]
            if tile == 3:
                if p.max_bombas < MAX_BOMBAS: p.max_bombas += 1
                mapa[p.grid_y][p.grid_x] = 0
                pontos[p.idx] += PONTOS_POWERUP_COLETADO
            elif tile == 4:
                if p.bomba_nivel < 4: p.bomba_nivel += 1
                mapa[p.grid_y][p.grid_x] = 0
                pontos[p.idx] += PONTOS_POWERUP_COLETADO

        vivos = [p for p in players if p.ativo]
        if len(vivos) <= 1:
            if len(vivos) == 1:
                pontos[vivos[0].idx] += PONTOS_VITORIA
            break

    vencedor = pontos.index(max(pontos))
    return vencedor, pontos


# ══════════════════════════════════════════════════════════════════════════════
#  LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

parar = False

def handler_ctrl_c(sig, frame):
    global parar
    parar = True

signal.signal(signal.SIGINT, handler_ctrl_c)

vitorias  = [0, 0, 0, 0]
partida   = 0
t_inicio  = time.time()
t_bloco   = time.time()   # tempo do último bloco de 50 partidas
vit_bloco = 0             # vitórias da IA1 no último bloco de 50

print(f"\n{'─'*55}")
print(f"  Iniciando treino... Ctrl+C para salvar e sair")
print(f"{'─'*55}\n")
print(f"{'Part':>6}  {'IA1 total':>9}  {'IA1 bloco':>9}  {'Placar':>20}  {'part/s':>6}")
print(f"{'─'*55}")

while not parar:
    vencedor, pontos = simular_partida()
    vitorias[vencedor] += 1
    partida += 1
    if vencedor == 0:
        vit_bloco += 1

    # Relatório a cada 50 partidas
    if partida % 50 == 0:
        agora      = time.time()
        dt_bloco   = agora - t_bloco
        taxa       = 50 / dt_bloco if dt_bloco > 0 else 0
        pct_total  = vitorias[0] / partida * 100
        pct_bloco  = vit_bloco / 50 * 100
        placar     = " ".join(f"{v}" for v in vitorias)

        print(f"{partida:>6}  "
              f"{vitorias[0]:>4} ({pct_total:4.1f}%)  "
              f"{vit_bloco:>4} ({pct_bloco:4.1f}%)  "
              f"{placar:>20}  "
              f"{taxa:>5.1f}")

        t_bloco   = agora
        vit_bloco = 0

# ── Encerramento ───────────────────────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"  Ctrl+C detectado — salvando Q-Table...")

# Força o save chamando direto o _salvar do módulo
try:
    ia1_mod._salvar()
    print(f"  Q-Table salva com sucesso!")
except Exception as e:
    print(f"  Erro ao salvar: {e}")

decorrido = time.time() - t_inicio
print(f"\n  Partidas jogadas : {partida}")
print(f"  Tempo total      : {decorrido:.0f}s  ({partida/decorrido:.1f} part/s)")
print(f"\n  Resultado final:")
for i, v in enumerate(vitorias):
    pct  = v / partida * 100 if partida else 0
    barra = "█" * int(pct / 2)
    print(f"    {nomes[i]:<15} {v:>5} vitórias ({pct:5.1f}%)  {barra}")
print(f"\n{'═'*55}\n")