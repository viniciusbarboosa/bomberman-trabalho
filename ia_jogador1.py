import json, os, random, collections
# VIDEO https://www.youtube.com/watch?v=tz8phEIqKAM
ALPHA         = 0.20
GAMMA         = 0.92
EPSILON_INI   = 1.0
EPSILON_MIN   = 0.08
EPSILON_DECAY = 0.9998
Q_TABLE_FILE  = "q_table_ia1.json"
MAX_Q_STATES  = 80_000

REPLAY_SIZE  = 10_000
REPLAY_BATCH = 32
REPLAY_EVERY = 4

# Só foge se bomba explode em <= 2s ou está no fogo
LIMIAR_FUGA  = 2.0

ACOES = ["cima", "baixo", "esquerda", "direita", "bomba", "parado"]

_G = {
    "q_table":   {},
    "epsilon":   EPSILON_INI,
    "passos":    0,
    "carregado": False,
    "s_ant":     None,
    "a_ant":     None,
    "pts_ant":   None,
    "vivo_ant":  True,
    "pos_ant":   None,
    "replay":    collections.deque(maxlen=REPLAY_SIZE),
}



def _carregar():
    if _G["carregado"]:
        return
    _G["carregado"] = True
    if os.path.exists(Q_TABLE_FILE):
        try:
            with open(Q_TABLE_FILE) as f:
                d = json.load(f)
            _G["q_table"] = d.get("q_table", {})
            _G["epsilon"] = d.get("epsilon", EPSILON_INI)
            _G["passos"]  = d.get("passos",  0)
            print(f"[IA-v3] Carregado: {len(_G['q_table'])} estados | "
                  f"e={_G['epsilon']:.3f} | passos={_G['passos']}")
        except Exception as e:
            print(f"[IA-v3] Erro ao carregar: {e}")


def _salvar():
    try:
        qt = _G["q_table"]
        if len(qt) > MAX_Q_STATES:
            chaves = list(qt.keys())
            random.shuffle(chaves)
            for k in chaves[:len(chaves) // 5]:
                del qt[k]
        with open(Q_TABLE_FILE, "w") as f:
            json.dump({"q_table": qt,
                       "epsilon": _G["epsilon"],
                       "passos":  _G["passos"]}, f)
    except Exception as e:
        print(f"[IA-v3] Erro ao salvar: {e}")


# ── Auxiliares ─────────────────────────────────────────────────────────────────
def _R(mapa): return len(mapa)
def _C(mapa): return len(mapa[0]) if mapa else 0


def _pode_mover(gx, gy, mapa, bombas):
    R, C = _R(mapa), _C(mapa)
    res = {}
    for nome, (dx, dy) in [("cima",(0,-1)),("baixo",(0,1)),
                             ("esquerda",(-1,0)),("direita",(1,0))]:
        nx, ny = gx+dx, gy+dy
        ok = (0<=nx<C and 0<=ny<R and mapa[ny][nx] in [0,3,4]
              and not any(b.x==nx and b.y==ny and not b.explodida for b in bombas))
        res[nome] = ok
    return res


def _tempo_bomba_ameacando(gx, gy, bombas):
    menor = 99.0
    for b in bombas:
        if b.explodida:
            continue
        alc = 1 + (b.nivel - 1) * 2
        if ((b.x == gx and abs(b.y - gy) <= alc) or
                (b.y == gy and abs(b.x - gx) <= alc)):
            menor = min(menor, b.tempo_explosao)
    return menor


def _no_fogo(gx, gy, bombas):
    return any(
        (gx, gy) in b.fogo
        for b in bombas if b.explodida and b.tempo_fogo > 0
    )


def _bfs_fuga(gx, gy, mapa, bombas):
    R, C = _R(mapa), _C(mapa)
    dirs = [("cima",0,-1),("baixo",0,1),("esquerda",-1,0),("direita",1,0)]
    fila = collections.deque()
    vis  = {(gx, gy)}
    for nome, dx, dy in dirs:
        nx, ny = gx+dx, gy+dy
        if not (0<=nx<C and 0<=ny<R): continue
        if mapa[ny][nx] not in [0,3,4]: continue
        if any(b.x==nx and b.y==ny and not b.explodida for b in bombas): continue
        fila.append((nx, ny, nome))
        vis.add((nx, ny))
    while fila:
        x, y, primeiro = fila.popleft()
        if _tempo_bomba_ameacando(x, y, bombas) > LIMIAR_FUGA and not _no_fogo(x, y, bombas):
            return primeiro
        for _, dx, dy in dirs:
            nx2, ny2 = x+dx, y+dy
            if (nx2,ny2) in vis: continue
            if not (0<=nx2<C and 0<=ny2<R): continue
            if mapa[ny2][nx2] not in [0,3,4]: continue
            if any(b.x==nx2 and b.y==ny2 and not b.explodida for b in bombas): continue
            vis.add((nx2,ny2))
            fila.append((nx2,ny2,primeiro))
    return None


def _blocos_ao_redor(gx, gy, mapa):
    R, C = _R(mapa), _C(mapa)
    return sum(
        1 for dx,dy in [(0,-1),(0,1),(-1,0),(1,0)]
        if 0<=gx+dx<C and 0<=gy+dy<R and mapa[gy+dy][gx+dx]==1
    )


def _inimigo_mp(self_player, players):
    melhor_d, melhor_p = 999, None
    for p in players:
        if p is self_player or not p.ativo: continue
        d = abs(self_player.grid_x-p.grid_x) + abs(self_player.grid_y-p.grid_y)
        if d < melhor_d:
            melhor_d, melhor_p = d, p
    return melhor_d, melhor_p


def _linha_visao(gx, gy, px, py, mapa):
    if gx == px:
        step = 1 if py>gy else -1
        for y in range(gy+step, py, step):
            if mapa[y][gx]==2: return False
        return True
    if gy == py:
        step = 1 if px>gx else -1
        for x in range(gx+step, px, step):
            if mapa[gy][x]==2: return False
        return True
    return False


# ── Estado ────────────────────────────────────────────────────────────────────

def _extrair_estado(self_player, mapa, players, bombas):
    gx, gy = self_player.grid_x, self_player.grid_y

    # 1) Nível de perigo: 0=ok 1=bomba vindo (2-3s) 2=urgente (<=2s) 3=fogo
    t_bomba = _tempo_bomba_ameacando(gx, gy, bombas)
    fogo    = _no_fogo(gx, gy, bombas)
    if fogo:              perigo = 3
    elif t_bomba <= LIMIAR_FUGA: perigo = 2
    elif t_bomba <= 3.0:  perigo = 1
    else:                 perigo = 0

    # 2) Urgência
    t_disc = 0 if t_bomba>3 else (1 if t_bomba>LIMIAR_FUGA else 2)

    # 3) Bitmask de direções livres
    livres = _pode_mover(gx, gy, mapa, bombas)
    bitmask = ((8 if livres["cima"]     else 0) |
               (4 if livres["baixo"]    else 0) |
               (2 if livres["esquerda"] else 0) |
               (1 if livres["direita"]  else 0))

    # 4) Distância ao inimigo
    dist_ini, ini_p = _inimigo_mp(self_player, players)
    dist_disc = 0 if dist_ini<=2 else (1 if dist_ini<=5 else (2 if dist_ini<=9 else 3))

    # 5) Blocos ao redor
    blocos = min(4, _blocos_ao_redor(gx, gy, mapa))

    # 6) Inimigo em linha de visão
    alinhado = int(bool(ini_p and _linha_visao(gx, gy, ini_p.grid_x, ini_p.grid_y, mapa)))

    # 7) Pode colocar bomba
    pode_bomba = int(len(self_player.bombas) < self_player.max_bombas)

    # 8) Powerup perto
    R, C = _R(mapa), _C(mapa)
    pu = int(any(
        mapa[gy+dy][gx+dx] in [3,4]
        for dy in range(-3,4) for dx in range(-3,4)
        if 0<=gx+dx<C and 0<=gy+dy<R
    ))

    # 9) Quadrante do inimigo
    quad = 4
    if ini_p:
        ddx = ini_p.grid_x - gx
        ddy = ini_p.grid_y - gy
        quad = (1 if ddx>0 else 0) if abs(ddx)>=abs(ddy) else (3 if ddy>0 else 2)

    return (perigo, t_disc, bitmask, dist_disc, blocos, alinhado, pode_bomba, pu, quad)


#QTable 
def _q(s):
    k = str(s)
    if k not in _G["q_table"]:
        _G["q_table"][k] = {a: 0.0 for a in ACOES}
    return _G["q_table"][k]


def _update(s, a, r, s2):
    qa = _q(s)
    qa[a] += ALPHA * (r + GAMMA * max(_q(s2).values()) - qa[a])


# ── Recompensa ────────────────────────────────────────────────────────────────

def _recompensa(self_player, players, bombas, mapa,
                pts_atual, pts_ant, ativo_atual, ativo_ant,
                pos_ant, acao_ant):
    r = 0.0
    gx, gy = self_player.grid_x, self_player.grid_y

    # Pontuação
    delta = pts_atual - pts_ant
    r += min(delta*0.003, 4.0) if delta>0 else max(delta*0.002, -1.0)

    # Morte
    if ativo_ant and not ativo_atual:
        r -= 10.0

    # Perigo atual
    t_bomba = _tempo_bomba_ameacando(gx, gy, bombas)
    fogo    = _no_fogo(gx, gy, bombas)
    if fogo:                    r -= 3.0
    elif t_bomba <= LIMIAR_FUGA: r -= 1.0
    elif t_bomba <= 3.0:         r -= 0.2

    # Fuga bem-sucedida
    if pos_ant:
        px, py = pos_ant
        t_ant = _tempo_bomba_ameacando(px, py, bombas)
        if (t_ant<=LIMIAR_FUGA or _no_fogo(px,py,bombas)) and t_bomba>LIMIAR_FUGA and not fogo:
            r += 2.0

    # Pressão agressiva
    dist_ini, ini_p = _inimigo_mp(self_player, players)
    if   dist_ini <= 2: r += 0.5
    elif dist_ini <= 4: r += 0.2
    elif dist_ini <= 7: r += 0.05
    else:               r -= 0.1

    # Bomba inteligente
    if acao_ant == "bomba":
        blocos   = _blocos_ao_redor(gx, gy, mapa)
        alinhado = ini_p and _linha_visao(gx, gy, ini_p.grid_x, ini_p.grid_y, mapa)
        r += 0.5 * blocos
        if alinhado:              r += 1.0
        if blocos==0 and not alinhado: r -= 0.8

    # Parado = ruim
    if acao_ant == "parado":
        r -= 0.2

    # Powerup perto
    R, C = _R(mapa), _C(mapa)
    if any(mapa[gy+dy][gx+dx] in [3,4]
           for dy in range(-3,4) for dx in range(-3,4)
           if 0<=gx+dx<C and 0<=gy+dy<R):
        r += 0.08

    return r


# ── Replay ────────────────────────────────────────────────────────────────────

def _replay_treinar():
    if len(_G["replay"]) < REPLAY_BATCH:
        return
    for s, a, r, s2 in random.sample(_G["replay"], REPLAY_BATCH):
        _update(s, a, r, s2)


# ── Política ──────────────────────────────────────────────────────────────────

def _escolher(s, self_player, mapa, bombas):
    gx, gy  = self_player.grid_x, self_player.grid_y
    livres  = _pode_mover(gx, gy, mapa, bombas)
    t_bomba = _tempo_bomba_ameacando(gx, gy, bombas)
    fogo    = _no_fogo(gx, gy, bombas)

    # Fuga SOMENTE quando urgente
    if fogo or t_bomba <= LIMIAR_FUGA:
        fuga = _bfs_fuga(gx, gy, mapa, bombas)
        if fuga:
            return fuga
        acoes_mv = [a for a in ["cima","baixo","esquerda","direita"] if livres.get(a)]
        return random.choice(acoes_mv) if acoes_mv else "parado"

    # Epsilon-greedy com viés agressivo
    validas = [a for a in ACOES if a in ("bomba","parado") or livres.get(a, False)]
    if not validas:
        validas = ACOES

    if random.random() < _G["epsilon"]:
        pesos = [0.3 if a=="parado" else 2.0 if a=="bomba" else 2.5 for a in validas]
        return random.choices(validas, weights=pesos)[0]
    else:
        qv = {a: _q(s)[a] for a in validas}
        return max(qv, key=qv.get)


# ── Função principal ──────────────────────────────────────────────────────────

def decidir_acao(self_player, mapa, players, bombas,
                 tempo_restante, pontos, hud_info, self_state):
    _carregar()

    idx = next((i for i, p in enumerate(players) if p is self_player), None)
    if idx is None:
        return "parado"

    pts_atual   = pontos[idx]
    ativo_atual = self_player.ativo
    s           = _extrair_estado(self_player, mapa, players, bombas)

    if _G["s_ant"] is not None:
        r = _recompensa(
            self_player, players, bombas, mapa,
            pts_atual, _G["pts_ant"],
            ativo_atual, _G["vivo_ant"],
            _G["pos_ant"], _G["a_ant"]
        )
        _update(_G["s_ant"], _G["a_ant"], r, s)
        _G["replay"].append((_G["s_ant"], _G["a_ant"], r, s))

    if _G["passos"] % REPLAY_EVERY == 0:
        _replay_treinar()

    acao = _escolher(s, self_player, mapa, bombas)

    _G["epsilon"] = max(EPSILON_MIN, _G["epsilon"] * EPSILON_DECAY)
    _G["passos"] += 1

    if _G["passos"] % 1000 == 0:
        _salvar()
        print(f"[IA-v3] passo={_G['passos']:>7} | "
              f"e={_G['epsilon']:.3f} | "
              f"estados={len(_G['q_table']):>6} | "
              f"replay={len(_G['replay']):>5}")

    _G["s_ant"]    = s
    _G["a_ant"]    = acao
    _G["pts_ant"]  = pts_atual
    _G["vivo_ant"] = ativo_atual
    _G["pos_ant"]  = (self_player.grid_x, self_player.grid_y)

    return acao