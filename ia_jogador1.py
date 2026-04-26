import random

ACOES = ["cima", "baixo", "esquerda", "direita", "bomba", "parado"]

_dir = {}
_fuga = {}
_passos = {}

def decidir_acao(player, mapa, jogadores, bombas, tempo_restante, pontos, hud_info, self_state):
    meu_x = self_state['grid_x']
    meu_y = self_state['grid_y']
    COLS = hud_info['cols']
    ROWS = hud_info['rows']
    pid = id(player)

    if pid not in _dir: _dir[pid] = random.choice(['baixo','direita'])
    if pid not in _fuga: _fuga[pid] = None
    if pid not in _passos: _passos[pid] = 0

    DIRS = [('cima',0,-1),('baixo',0,1),('esquerda',-1,0),('direita',1,0)]

    def livre(x, y):
        if not (0 <= x < COLS and 0 <= y < ROWS): return False
        if mapa[y][x] not in (0, 3, 4): return False
        if any(b.x == x and b.y == y and not b.explodida for b in bombas): return False
        return True

    def bloco(x, y):
        if not (0 <= x < COLS and 0 <= y < ROWS): return False
        return mapa[y][x] == 1

    def em_perigo(x, y):
        for b in bombas:
            if not b.explodida:
                if b.x == x and b.y == y: return True
                if b.x == x and abs(b.y - y) <= b.nivel * 2: return True
                if b.y == y and abs(b.x - x) <= b.nivel * 2: return True
            elif (x, y) in b.fogo: return True
        return False

    def segura(x, y):
        return livre(x, y) and not em_perigo(x, y)

    #SE TIVER BOMBA ELE FOGE SEMPRE
    if em_perigo(meu_x, meu_y):
        opcoes = [n for n,dx,dy in DIRS if segura(meu_x+dx, meu_y+dy)]
        if not opcoes:
            opcoes = [n for n,dx,dy in DIRS if livre(meu_x+dx, meu_y+dy)]
        if opcoes:
            #prefere continuar na fuga atual
            if _fuga[pid] and _fuga[pid] in opcoes:
                return _fuga[pid]
            _fuga[pid] = random.choice(opcoes)
            _passos[pid] = 4  #fuga evita bug verificar
            return _fuga[pid]
        return 'parado'

    #se tiver passos na fuga ele continua fugindo
    if _passos[pid] > 0:
        _passos[pid] -= 1
        if _fuga[pid]:
            for nome, dx, dy in DIRS:
                if nome == _fuga[pid] and livre(meu_x+dx, meu_y+dy):
                    return _fuga[pid]
        _fuga[pid] = None

    #saiu do perigo termina fugir
    if not em_perigo(meu_x, meu_y) and _passos[pid] == 0:
        _fuga[pid] = None

    #se tiver boloco na frente plata bobma
    for nome, dx, dy in DIRS:
        if nome == _dir[pid]:
            nx, ny = meu_x+dx, meu_y+dy
            if bloco(nx, ny) and self_state['bombas_ativas'] < self_state['max_bombas']:
                fugas = [n for n,fdx,fdy in DIRS if n != nome and segura(meu_x+fdx, meu_y+fdy)]
                if fugas:
                    _fuga[pid] = random.choice(fugas)
                    _passos[pid] = 4
                    return 'bomba'

    #inimigo perto temq ue plantar a bomba
    inimigos = [p for p in jogadores if p is not player and p.ativo]
    if inimigos:
        alvo = min(inimigos, key=lambda p: abs(p.grid_x-meu_x)+abs(p.grid_y-meu_y))
        dist = abs(alvo.grid_x-meu_x) + abs(alvo.grid_y-meu_y)
        if dist <= 2 and self_state['bombas_ativas'] < self_state['max_bombas']:
            fugas = [n for n,dx,dy in DIRS if segura(meu_x+dx, meu_y+dy)]
            if fugas:
                _fuga[pid] = random.choice(fugas)
                _passos[pid] = 4
                return 'bomba'

        #persegue inimigo , verificar a logica com a da direção do inimigo pq ta bugando
        dx = alvo.grid_x - meu_x
        dy = alvo.grid_y - meu_y
        nova = ('direita' if dx > 0 else 'esquerda') if abs(dx) >= abs(dy) else ('baixo' if dy > 0 else 'cima')
        for nome, ddx, ddy in DIRS:
            if nome == nova and livre(meu_x+ddx, meu_y+ddy):
                _dir[pid] = nova
                return nova

    #mantem direçãoa tual para fugir de bombas
    for nome, dx, dy in DIRS:
        if nome == _dir[pid] and livre(meu_x+dx, meu_y+dy):
            return _dir[pid]

    #boloqueado ele vai plantar
    if self_state['bombas_ativas'] < self_state['max_bombas']:
        fugas = [n for n,dx,dy in DIRS if segura(meu_x+dx, meu_y+dy)]
        if fugas:
            _fuga[pid] = random.choice(fugas)
            _passos[pid] = 4
            return 'bomba'

    return 'parado'