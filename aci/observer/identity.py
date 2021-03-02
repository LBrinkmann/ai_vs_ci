class IdentityView():
    def __init__(self, *_, **_):
        pass

    def __call__(self, **state):
        return state
