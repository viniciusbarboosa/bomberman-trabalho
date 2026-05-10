import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ── CONFIGURAÇÕES DA REDE (UNIFICADO) ───────────────────────────────────────

INPUT_DIM = 9
ACOES = ["cima", "baixo", "esquerda", "direita", "bomba", "parado"]


class BombermanNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.camadas = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, len(ACOES))
        )

    def forward(self, x):
        return self.camadas(x)


class AgenteRL:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.modelo = BombermanNet().to(self.device)

        self.otimizador = optim.Adam(
            self.modelo.parameters(),
            lr=0.0003
        )

        self.epsilon = 0.2

        if os.path.exists("ia_neural.pth"):
            try:
                self.modelo.load_state_dict(
                    torch.load(
                        "ia_neural.pth",
                        map_location=self.device
                    )
                )
            except:
                pass

    def decidir(self, estado, proibidas=None):
        if random.random() < self.epsilon:
            validas = [
                i for i in range(6)
                if i not in (proibidas or [])
            ]

            return random.choice(validas) if validas else 5

        with torch.no_grad():
            q = self.modelo(
                torch.FloatTensor(estado).to(self.device)
            ).cpu().numpy()

            if proibidas:
                for p in proibidas:
                    q[p] = -1e9

            return np.argmax(q)

    def treinar(self, s, a, r, s2, done):
        s = torch.FloatTensor(s).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)

        alvo = (
            r +
            0.95 * self.modelo(s2).max() * (1 - int(done))
        )

        loss = nn.MSELoss()(
            self.modelo(s)[a],
            alvo.detach()
        )

        self.otimizador.zero_grad()
        loss.backward()
        self.otimizador.step()


_brain = AgenteRL()

_G = {
    "s_ant": None,
    "a_ant": None,
    "pts_ant": 0
}

# ── LÓGICA DE PERIGO E BUSCA DE ROTA ────────────────────────────────────────


def _obter_perigo(mapa, bombas):
    perigo = set()

    for b in bombas:
        perigo.add((b.x, b.y))

        alcance = 1 + (b.nivel - 1) * 2

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for i in range(1, alcance + 1):

                nx = b.x + dx * i
                ny = b.y + dy * i

                if (
                    not (0 <= nx < len(mapa[0]) and 0 <= ny < len(mapa))
                    or mapa[ny][nx] == 2
                ):
                    break

                perigo.add((nx, ny))

                if mapa[ny][nx] == 1:
                    break

    return perigo


def _buscar_fuga(gx, gy, mapa, perigo):
    fila = [(gx, gy, [])]
    visitados = {(gx, gy)}

    while fila:
        cx, cy, path = fila.pop(0)

        if (cx, cy) not in perigo:
            return path[0] if path else 5

        for i, (dx, dy) in enumerate([
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0)
        ]):

            nx = cx + dx
            ny = cy + dy

            if 0 <= nx < len(mapa[0]) and 0 <= ny < len(mapa):

                if (
                    mapa[ny][nx] in [0, 3, 4]
                    and (nx, ny) not in visitados
                ):
                    visitados.add((nx, ny))
                    fila.append((nx, ny, path + [i]))

    return 5


# ── EXTRAÇÃO DE ESTADO ───────────────────────────────────────────────────────


def _extrair_estado_completo(
    p,
    mapa,
    players,
    bombas,
    perigo
):
    gx, gy = p.grid_x, p.grid_y

    tx, ty, dist = gx, gy, 99

    # 1. FOCO NO PLAYER
    for o in players:
        if o is not p and o.ativo:

            d = abs(gx - o.grid_x) + abs(gy - o.grid_y)

            if d < dist:
                tx, ty, dist = o.grid_x, o.grid_y, d

    # 2. FOCO EM BLOCOS
    if dist == 99:

        for y, linha in enumerate(mapa):
            for x, valor in enumerate(linha):

                if valor == 1:

                    d = abs(gx - x) + abs(gy - y)

                    if d < dist:
                        tx, ty, dist = x, y, d

    # Mobilidade
    saidas = sum(
        1
        for dx, dy in [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0)
        ]
        if (
            0 <= gx + dx < len(mapa[0])
            and 0 <= gy + dy < len(mapa)
            and mapa[gy + dy][gx + dx] in [0, 3, 4]
        )
    )

    return (
        1 if (gx, gy) in perigo else 0,

        1 if tx > gx else (
            -1 if tx < gx else 0
        ),

        1 if ty > gy else (
            -1 if ty < gy else 0
        ),

        dist / 15.0,

        1 if len(p.bombas) < p.max_bombas else 0,

        gx / 13.0,
        gy / 11.0,

        saidas / 4.0,

        len(bombas) / 5.0
    )


# ── DECISÃO FINAL UNIFICADA ─────────────────────────────────────────────────


def decidir_acao(
    self_player,
    mapa,
    players,
    bombas,
    tempo,
    pontos,
    hud,
    self_state
):
    global _G

    idx = (
        players.index(self_player)
        if self_player in players
        else 0
    )

    gx, gy = self_player.grid_x, self_player.grid_y

    perigo = _obter_perigo(mapa, bombas)

    s_atu = _extrair_estado_completo(
        self_player,
        mapa,
        players,
        bombas,
        perigo
    )

    # 1. TREINO
    if _G["s_ant"] is not None:

        recompensa = 0.1

        if not self_player.ativo:
            recompensa = -15000.0

        elif pontos[idx] > _G["pts_ant"]:
            recompensa = 3000.0

        elif (gx, gy) in perigo:
            recompensa = -500.0

        _brain.treinar(
            _G["s_ant"],
            _G["a_ant"],
            recompensa,
            s_atu,
            not self_player.ativo
        )

    # 2. LÓGICA HÍBRIDA
    if (gx, gy) in perigo:

        a_idx = _buscar_fuga(
            gx,
            gy,
            mapa,
            perigo
        )

    else:
        # IA decide ataque
        proibidas = []

        if len(self_player.bombas) >= self_player.max_bombas:
            proibidas.append(4)

        # Simulação de explosão futura
        perigo_futuro = perigo.copy()

        perigo_futuro.add((gx, gy))

        for dx, dy in [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0)
        ]:

            for i in range(1, 3):

                nx = gx + dx * i
                ny = gy + dy * i

                if (
                    not (
                        0 <= nx < len(mapa[0])
                        and 0 <= ny < len(mapa)
                    )
                    or mapa[ny][nx] in [1, 2]
                ):
                    break

                perigo_futuro.add((nx, ny))

        # Bloqueia bomba se não houver fuga
        if _buscar_fuga(
            gx,
            gy,
            mapa,
            perigo_futuro
        ) == 5:

            proibidas.append(4)

        a_idx = _brain.decidir(
            s_atu,
            proibidas=proibidas
        )

    # 3. FILTRO ANTI-RETORNO
    nx, ny = gx, gy

    if ACOES[a_idx] == "cima":
        ny -= 1

    elif ACOES[a_idx] == "baixo":
        ny += 1

    elif ACOES[a_idx] == "esquerda":
        nx -= 1

    elif ACOES[a_idx] == "direita":
        nx += 1

    if (
        (nx, ny) in perigo
        and ACOES[a_idx] != "parado"
    ):
        a_idx = _buscar_fuga(
            gx,
            gy,
            mapa,
            perigo
        )

    _G["s_ant"] = s_atu
    _G["a_ant"] = a_idx
    _G["pts_ant"] = pontos[idx]

    return ACOES[a_idx]


def _salvar():
    torch.save(
        _brain.modelo.state_dict(),
        "ia_neural.pth"
    )