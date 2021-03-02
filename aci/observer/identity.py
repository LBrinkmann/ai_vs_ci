class IdentityView():
    shape = None

    def __init__(self, *_, **__):
        pass

    def __call__(self, **state):
        return state
