from nltk.tree import Tree, ParentedTree, ImmutableTree

def parse_and_binarise(tree_string):
    tree = Tree.fromstring(tree_string)
    Tree.chomsky_normal_form(tree, childChar='-')
    return RHTree.convert(tree)

class RHTree(ParentedTree):
    def rh_distances(self):
        rh_distances = []
        h_r = self.root().height() # NLTK tree nodes have height 2
        leaf_num = len(self.leaves())
        for i, leaf in enumerate(self.leaves()):
            rh_distances.append(
                (self.h_i(i, leaf_num) - 2) / (h_r - 1))
        return rh_distances

    def h_i(self, i, leaf_num):
        # h_i = h_r + delta_h_i
        if i == 0:
            return self.root().height() + 1

        treeposition = self.treeposition_spanning_leaves(i-1, i+1)
        t = self.root()
        for tp in treeposition:
            t = t[tp]
        return t.height()

    def base_label(self):
        return self.label().split('-')[0].split('=')[0]

    def tree_sequence(self):
        seq = ['(', self.base_label()]
        for child in self:
            if isinstance(child, RHTree):
                seq += child.tree_sequence()
        seq += [')']
        return seq

    def pformat_tree_sequence(self, seq=None, tab='\t'):
        formatted = ''

        if seq is None:
            seq = self.tree_sequence()

        indent = -1
        for t in seq:
            if t == '(':
                indent += 1
                formatted += '\n' + tab * indent + '('
            elif t == ')':
                indent -= 1
                formatted += ')'
            else:
                formatted += t
        return formatted

if __name__ == '__main__':
    tree = '(S (NP (NP (PN Buffalo) (N buffalo)) (RC (NP (PN Buffalo) (N buffalo)) (V buffalo))) (VP (V buffalo) (NP (PN Buffalo) (N buffalo))))'
    tree = RHTree.fromstring(tree)
    d = tree.rh_distances()

    assert d[0] == 1
    assert d[2] == 3/5
    assert d[4] == 2/5


    t2 = RHTree.fromstring('(S (NP (PRP$ my) (NN dog)) (VP (VBZ is) (ADJP (JJ cute))))')
    d2 = t2.rh_distances()