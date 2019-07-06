import pytest

# TODO: input train range
class InsufficientAmount(Exception):
    pass


class Wallet:
    def __init__(self, amount=0):
        self.balance = amount

    def add_cash(self, amount):
        self.balance += amount

    def spend_cash(self, amount):
        if self.balance < amount:
            raise InsufficientAmount("Exception (InsufficientAmount): Balance {} < amount {}."
                                     .format(self.balance, amount))
        self.balance -= amount

@pytest.fixture
def empty_wallet():
    """
    Instantiates an empty wallet.
    :return: An empty Wallet object.
    """
    return Wallet()



def test_default_initial_amount(empty_wallet):
    """
    Checks if an empty wallet object has a balance of zero.
    :param empty_wallet: A wallet object.
    """
    assert empty_wallet.balance == 0

