class Node:
    def __init__(self, value):
        self.value = value

    # Lets you compare class objects
    def __eq__(self, other) -> bool:
        if not isinstance(other, Node):         # not Node object
            return False
        return self.value == other.value        # check if they have the same value

    # allows Node to be a key in a dict
    def __hash__(self):
        return hash(self.value)

    # Prints the value
    def __str__(self):
        return str(self.value)
