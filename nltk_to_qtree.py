def _compute_repr(tree):
    """
    Helper function to be called recursively in constructing the qtree
    representation
    """
    if tree.height() == 2:
        return "[.{} {} ] ".format(tree.label(), "".join(tree.leaves()))
    else:
        s = ""
        for child in tree:
            s += _compute_repr(child)
        return "[.{} {} ] ".format(tree.label(), s)

def nltk_to_qtree(tree):
    """
    Takes an nltk.tree.Tree and returns a string that can be used to draw
    it with qtree
    """
    return "\\"+"Tree {}".format(_compute_repr(tree))
    
        