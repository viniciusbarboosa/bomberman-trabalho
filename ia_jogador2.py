import os
import random
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES
# ══════════════════════════════════════════════════════════════════════════════

INPUT_DIM  = 30
ACOES      = ["cima", "baixo", "esquerda", "direita", "bomba", "parado"]
N_ACOES    = len(ACOES)

# Constantes de pontos (espelham o jogo)
PONTOS_BLOCO             = 100
PONTOS_POWERUP_COLETADO  = 200
PONTOS_POWERUP_DESTRUIDO = -50
PONTOS_MATAR_JOGADOR     = 1000
PONTOS_VITORIA           = 10000

# Hiperparâmetros DQN
GAMMA         = 0.97
LR            = 0.0002
BATCH_SIZE    = 512
BUFFER_SIZE   = 100_000
MIN_BUFFER    = 2_000
TARGET_UPDATE = 300
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.9995

MODEL_PATH = "ia_neural2.pth"

# ══════════════════════════════════════════════════════════════════════════════
#  REDE NEURAL — Dueling DQN
#  V(s) + A(s,a) - mean(A) = Q(s,a)
# ══════════════════════════════════════════════════════════════════════════════

class BombermanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.value     = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, N_ACOES)

    def forward(self, x):
        h = self.trunk(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + a - a.mean(dim=-1, keepdim=True)


# ══════════════════════════════════════════════════════════════════════════════
#  REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, float(r), s2, float(done)))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(s2)),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buf)


# ══════════════════════════════════════════════════════════════════════════════
#  AGENTE DQN
# ══════════════════════════════════════════════════════════════════════════════

class AgenteRL:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online = BombermanNet().to(self.device)
        self.target = BombermanNet().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.otimizador  = optim.Adam(self.online.parameters(), lr=LR)
        self.scheduler   = optim.lr_scheduler.StepLR(self.otimizador, step_size=10_000, gamma=0.5)
        self.buffer      = ReplayBuffer()
        self.epsilon     = EPSILON_START
        self.passo_total = 0

        if os.path.exists(MODEL_PATH):
            try:
                ckpt = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
                if isinstance(ckpt, dict) and "online" in ckpt:
                    try:
                        self.online.load_state_dict(ckpt["online"])
                        self.target.load_state_dict(ckpt["target"])
                        self.epsilon     = ckpt.get("epsilon", EPSILON_END)
                        self.passo_total = ckpt.get("passo_total", 0)
                        print(f"[IA] Modelo carregado. e={self.epsilon:.3f} | passos={self.passo_total}")
                    except RuntimeError:
                        print("[IA] Arquitetura mudou — apague ia_neural.pth e retreine.")
                else:
                    print("[IA] Formato antigo ignorado — iniciando do zero.")
            except Exception as e:
                print(f"[IA] Erro ao carregar: {e}")

    def decidir(self, estado, proibidas=None):
        proibidas = proibidas or []
        if random.random() < self.epsilon:
            validas = [i for i in range(N_ACOES) if i not in proibidas]
            return random.choice(validas) if validas else 5

        with torch.no_grad():
            q = self.online(
                torch.FloatTensor(estado).unsqueeze(0).to(self.device)
            ).squeeze().cpu().numpy()

        for p in proibidas:
            q[p] = -1e9
        return int(np.argmax(q))

    def armazenar(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def treinar_batch(self):
        if len(self.buffer) < MIN_BUFFER:
            return
        self.passo_total += 1

        s, a, r, s2, d = self.buffer.sample(BATCH_SIZE)
        s  = s.to(self.device)
        a  = a.to(self.device)
        r  = r.to(self.device)
        s2 = s2.to(self.device)
        d  = d.to(self.device)

        with torch.no_grad():
            a_best = self.online(s2).argmax(dim=1)
            q_next = self.target(s2).gather(1, a_best.unsqueeze(1)).squeeze()
            alvo   = r + GAMMA * q_next * (1 - d)

        q_pred = self.online(s).gather(1, a.unsqueeze(1)).squeeze()
        loss   = nn.SmoothL1Loss()(q_pred, alvo)

        self.otimizador.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.otimizador.step()
        self.scheduler.step()

        if self.passo_total % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

    def salvar(self):
        torch.save({
            "online":      self.online.state_dict(),
            "target":      self.target.state_dict(),
            "epsilon":     self.epsilon,
            "passo_total": self.passo_total,
        }, MODEL_PATH)
        print(f"[IA] Salvo. e={self.epsilon:.4f} | passos={self.passo_total}")


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAIS
# ══════════════════════════════════════════════════════════════════════════════

_brain   = AgenteRL()
_estados = {}


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE MAPA
# ══════════════════════════════════════════════════════════════════════════════

def _obter_perigo(mapa, bombas):
    perigo = set()
    for b in bombas:
        perigo.add((b.x, b.y))
        alcance = 1 + (b.nivel - 1) * 2
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            for i in range(1, alcance + 1):
                nx, ny = b.x + dx*i, b.y + dy*i
                if not (0 <= nx < len(mapa[0]) and 0 <= ny < len(mapa)):
                    break
                if mapa[ny][nx] == 2:
                    break
                perigo.add((nx, ny))
                if mapa[ny][nx] == 1:
                    break
    return perigo


def _buscar_fuga(gx, gy, mapa, perigo):
    fila     = [(gx, gy, [])]
    visitado = {(gx, gy)}
    DIRS     = [(0,-1),(0,1),(-1,0),(1,0)]
    while fila:
        cx, cy, path = fila.pop(0)
        if (cx, cy) not in perigo:
            return path[0] if path else 5
        for i, (dx, dy) in enumerate(DIRS):
            nx, ny = cx+dx, cy+dy
            if (0 <= nx < len(mapa[0]) and 0 <= ny < len(mapa)
                    and mapa[ny][nx] in [0, 3, 4]
                    and (nx, ny) not in visitado):
                visitado.add((nx, ny))
                fila.append((nx, ny, path + [i]))
    return 5


def _blocos_em_linha(gx, gy, mapa, nivel):
    """Quantos blocos quebráveis seriam atingidos por bomba colocada em (gx,gy)."""
    COLS_MAP = len(mapa[0])
    ROWS_MAP = len(mapa)
    alcance  = 1 + (nivel - 1) * 2
    count    = 0
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        for i in range(1, alcance + 1):
            nx, ny = gx + dx*i, gy + dy*i
            if not (0 <= nx < COLS_MAP and 0 <= ny < ROWS_MAP) or mapa[ny][nx] == 2:
                break
            if mapa[ny][nx] == 1:
                count += 1
                break
    return count


def _inimigo_em_linha(gx, gy, mapa, players, self_player, nivel):
    """1 se inimigo está na linha de fogo de bomba colocada em (gx,gy)."""
    COLS_MAP = len(mapa[0])
    ROWS_MAP = len(mapa)
    alcance  = 1 + (nivel - 1) * 2
    inimigos = {(o.grid_x, o.grid_y) for o in players if o is not self_player and o.ativo}
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        for i in range(1, alcance + 1):
            nx, ny = gx + dx*i, gy + dy*i
            if not (0 <= nx < COLS_MAP and 0 <= ny < ROWS_MAP) or mapa[ny][nx] == 2:
                break
            if (nx, ny) in inimigos:
                return 1
            if mapa[ny][nx] == 1:
                break
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRAÇÃO DE ESTADO — 30 features
# ══════════════════════════════════════════════════════════════════════════════

def _extrair_estado(p, mapa, players, bombas, perigo):
    gx, gy   = p.grid_x, p.grid_y
    COLS_MAP = len(mapa[0])
    ROWS_MAP = len(mapa)
    DIRS4    = [(0,-1),(0,1),(-1,0),(1,0)]

    # Inimigo mais próximo
    tx, ty, dist_ini = gx, gy, 99.0
    for o in players:
        if o is not p and o.ativo:
            d = abs(gx - o.grid_x) + abs(gy - o.grid_y)
            if d < dist_ini:
                tx, ty, dist_ini = o.grid_x, o.grid_y, float(d)

    # Bloco quebrável mais próximo
    bx, by, dist_bloco = gx, gy, 99.0
    for y, linha in enumerate(mapa):
        for x, v in enumerate(linha):
            if v == 1:
                d = abs(gx - x) + abs(gy - y)
                if d < dist_bloco:
                    bx, by, dist_bloco = x, y, float(d)

    # Saídas livres
    saidas = sum(
        1 for dx, dy in DIRS4
        if 0 <= gx+dx < COLS_MAP and 0 <= gy+dy < ROWS_MAP
        and mapa[gy+dy][gx+dx] in [0, 3, 4]
    )

    # Vizinhança por tipo — diferencia quebrável de sólida
    viz_livre     = []
    viz_quebravel = []
    viz_solida    = []
    viz_perigo    = []
    for dx, dy in DIRS4:
        nx, ny  = gx+dx, gy+dy
        em_mapa = 0 <= nx < COLS_MAP and 0 <= ny < ROWS_MAP
        val     = mapa[ny][nx] if em_mapa else 2
        viz_livre.append(    1.0 if em_mapa and val in [0, 3, 4] else 0.0)
        viz_quebravel.append(1.0 if em_mapa and val == 1          else 0.0)
        viz_solida.append(   1.0 if (not em_mapa) or val == 2     else 0.0)
        viz_perigo.append(   1.0 if em_mapa and (nx, ny) in perigo else 0.0)

    # Bomba mais próxima
    dist_bomba  = 1.0
    tempo_bomba = 1.0
    for b in bombas:
        if not b.explodida:
            d = (abs(gx - b.x) + abs(gy - b.y)) / 15.0
            if d < dist_bomba:
                dist_bomba  = d
                tempo_bomba = min(b.tempo_explosao / 4.0, 1.0)

    # Powerup mais próximo
    dist_pu = 1.0
    for y, linha in enumerate(mapa):
        for x, v in enumerate(linha):
            if v in [3, 4]:
                d = (abs(gx - x) + abs(gy - y)) / 20.0
                dist_pu = min(dist_pu, d)

    # Utilidade de bomba aqui
    blocos_atingiveis = _blocos_em_linha(gx, gy, mapa, p.bomba_nivel)

    estado = [
        # Perigo e mobilidade (4)
        1.0 if (gx, gy) in perigo else 0.0,             # 0
        saidas / 4.0,                                    # 1
        dist_bomba,                                      # 2
        tempo_bomba,                                     # 3

        # Posição (2)
        gx / (COLS_MAP - 1),                             # 4
        gy / (ROWS_MAP - 1),                             # 5

        # Inimigo mais próximo (3)
        1.0 if tx > gx else (-1.0 if tx < gx else 0.0), # 6
        1.0 if ty > gy else (-1.0 if ty < gy else 0.0), # 7
        min(dist_ini, 20.0) / 20.0,                      # 8

        # Bloco quebrável mais próximo (3)
        1.0 if bx > gx else (-1.0 if bx < gx else 0.0), # 9
        1.0 if by > gy else (-1.0 if by < gy else 0.0), # 10
        min(dist_bloco, 20.0) / 20.0,                    # 11

        # Vizinhança livre (4)
        *viz_livre,                                      # 12-15

        # Vizinhança quebrável (4)
        *viz_quebravel,                                  # 16-19

        # Vizinhança sólida (4)
        *viz_solida,                                     # 20-23

        # Vizinhança perigo (4)
        *viz_perigo,                                     # 24-27

        # Extras (2)
        dist_pu,                                         # 28
        blocos_atingiveis / 4.0,                         # 29
    ]

    assert len(estado) == INPUT_DIM, f"Estado com {len(estado)} features!"
    return tuple(float(x) for x in estado)


# ══════════════════════════════════════════════════════════════════════════════
#  RECOMPENSA
# ══════════════════════════════════════════════════════════════════════════════

def _calcular_recompensa(p, idx, gx, gy, pontos, perigo, prev, mapa, players):
    if not p.ativo:
        return -600.0

    r = 0.0
    delta_pts = pontos[idx] - prev["pts_ant"]

    # Eventos de pontuação
    if delta_pts >= PONTOS_MATAR_JOGADOR:
        r += 400.0
    elif delta_pts >= PONTOS_POWERUP_COLETADO:
        r += 25.0
    elif delta_pts >= PONTOS_BLOCO:
        r += 8.0

    # Perigo e parado
    if (gx, gy) in perigo:
        r -= 40.0
    if prev["pos_ant"] == (gx, gy):
        r -= 3.0

    # Caçar inimigo
    inimigos = [o for o in players if o is not p and o.ativo]
    if inimigos:
        dist_agora = min(abs(gx - o.grid_x) + abs(gy - o.grid_y) for o in inimigos)
        dist_antes = min(
            abs(prev["pos_ant"][0] - o.grid_x) + abs(prev["pos_ant"][1] - o.grid_y)
            for o in inimigos
        )
        if dist_agora < dist_antes:
            r += 2.0
        elif dist_agora > dist_antes:
            r -= 1.0

    # Bomba estratégica
    if prev.get("a_ant") == 4:
        px, py = prev["pos_ant"]
        if _inimigo_em_linha(px, py, mapa, players, p, p.bomba_nivel):
            r += 50.0
        blocos = _blocos_em_linha(px, py, mapa, p.bomba_nivel)
        r += blocos * 3.0

    # Sobrevivência
    vivos = sum(1 for pl in players if pl.ativo)
    if vivos == 1 and p.ativo:
        r += 300.0
    else:
        r += 0.1

    return float(np.clip(r, -700.0, 500.0))


# ══════════════════════════════════════════════════════════════════════════════
#  FUNÇÃO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def decidir_acao(self_player, mapa, players, bombas, tempo, pontos, hud, self_state):
    try:
        idx = players.index(self_player)
    except ValueError:
        idx = 0

    gx, gy = self_player.grid_x, self_player.grid_y
    perigo = _obter_perigo(mapa, bombas)
    s_atu  = _extrair_estado(self_player, mapa, players, bombas, perigo)

    # Treina com experiência anterior
    if idx in _estados and _estados[idx].get("s_ant") is not None:
        prev = _estados[idx]
        r = _calcular_recompensa(self_player, idx, gx, gy, pontos, perigo, prev, mapa, players)
        _brain.armazenar(prev["s_ant"], prev["a_ant"], r, s_atu, not self_player.ativo)
        _brain.treinar_batch()

    if not self_player.ativo:
        _estados[idx] = {"s_ant": None, "a_ant": None, "pts_ant": pontos[idx], "pos_ant": (gx, gy)}
        return "parado"

    # Fuga prioritária
    if (gx, gy) in perigo:
        a_idx = _buscar_fuga(gx, gy, mapa, perigo)
    else:
        proibidas = []

        if len(self_player.bombas) >= self_player.max_bombas:
            proibidas.append(4)
        else:
            # Bloqueia bomba se não houver fuga
            perigo_futuro = set(perigo)
            perigo_futuro.add((gx, gy))
            alcance = 1 + (self_player.bomba_nivel - 1) * 2
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                for i in range(1, alcance + 1):
                    nx, ny = gx + dx*i, gy + dy*i
                    if not (0 <= nx < len(mapa[0]) and 0 <= ny < len(mapa)):
                        break
                    if mapa[ny][nx] in [1, 2]:
                        break
                    perigo_futuro.add((nx, ny))
            if _buscar_fuga(gx, gy, mapa, perigo_futuro) == 5:
                proibidas.append(4)

        a_idx = _brain.decidir(s_atu, proibidas=proibidas)

    # Filtro final: não andar para perigo
    nx, ny = gx, gy
    if ACOES[a_idx] == "cima":       ny -= 1
    elif ACOES[a_idx] == "baixo":    ny += 1
    elif ACOES[a_idx] == "esquerda": nx -= 1
    elif ACOES[a_idx] == "direita":  nx += 1

    if (nx, ny) in perigo and ACOES[a_idx] not in ["bomba", "parado"]:
        a_idx = _buscar_fuga(gx, gy, mapa, perigo)

    _estados[idx] = {
        "s_ant":   s_atu,
        "a_ant":   a_idx,
        "pts_ant": pontos[idx],
        "pos_ant": (gx, gy),
    }

    return ACOES[a_idx]


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS PARA O TREINADOR
# ══════════════════════════════════════════════════════════════════════════════

def decay_epsilon():
    _brain.epsilon = max(EPSILON_END, _brain.epsilon * EPSILON_DECAY)

def salvar():
    _brain.salvar()