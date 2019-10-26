class ImageTreeNode:
    def __init__(self, name, parent, label=None, data=None):
        self.name = name
        self.label = label
        self.parent = parent
        self.mask = data
        self.depth = -1 #nível da arvore
        self.neighbors = -1 #quantidade de nós no mesmo nível
        self.childs = []

    def add_child(self, child):
        self.childs.append(child)

    def remove_child(self, child):
        self.childs.remove(child)

    def set_label(self, label):
        self.label = label
    
    def print_node(self, indent=''):
        print(indent, "+ TreeNode(", self.name, "):")
        print(indent, "\t- label = ", 
            ("Unlabeled" if self.label is None else self.label))
        print(indent, "\t- depth = ", str(self.depth))
        print(indent, "\t- neighbors count = ", str(self.neighbors))
        print(indent, "\t- parent =", (self.parent.name, self.parent)
            if(self.parent) else 'None')
        print(indent, "\t- mask =", self.mask)
        print(indent, "\t- childs =", [(c.name, c) for c in self.childs])
        print()

class ImageTree:
    def __init__(self, hierarchy, masks, name='-1'):
        self.root = ImageTreeNode(name='-1', parent=None,
            label=None, data=None)
        self.total_nodes = 1#Setado em 1 pq tem a raiz
                            #Atributo alterado em: _build_subtree;
                            # _cut_off; _cut_off_none_child
        self._build_tree(hierarchy, masks)
        self.name = name
        self.label = None

    def _build_subtree(self, hierarchy, parent, index, 
        brothers_list=[]):
        node = ImageTreeNode(name=str(index), parent=parent, 
            label=None, data=hierarchy[index][5])
        self.total_nodes += 1
        if(hierarchy[index][2] == -1):
            if(hierarchy[index][0] == -1):
                return node
            elif(hierarchy[index][1] == -1): #first leaf
                brothers_list.append(node)
                bi = hierarchy[index][0]
                while(bi != -1 and parent != self.root):
                    k = self._build_subtree(hierarchy, parent,
                        bi, brothers_list)
                    brothers_list.append(k)
                    bi = hierarchy[bi][0]

                return brothers_list
            return node
        else:
            inf_tree = self._build_subtree(hierarchy, node,
                hierarchy[index][2], [])
            if(isinstance(inf_tree, list)):
                for st in inf_tree: node.add_child(st)
            else: node.add_child(inf_tree)

        return node

    def _build_tree(self, hierarchy, masks):
        '''
        hierarchy[i][0] = Next; .[1] = Previous; .[2] = First Child;
        hierarchy[i][3] = Parent; .[4] = Usable (1=T|0=F); .[5] = Mask
        '''

        def _merge_hierarchy_mask(hierarchy, masks):
            h = []
            mask_index = 0
            for i in range(len(hierarchy)):
                element = hierarchy[i].tolist()
                if(element[4] == 1):
                    element.append(masks[mask_index])
                    mask_index += 1
                else:
                    element.append(None)
                h.append(element)

            return h

        def _build_stack(hierarchy):
            stack = []
            for i in range(len(hierarchy)):
                if(hierarchy[i][3] == -1):
                    stack.append(i)

            return stack

        def _cut_off_none_child(node):
            for child in list(node.childs):
                if child.mask is None: 
                    node.childs.remove(child)
                    self.total_nodes -= 1
                else:
                    _cut_off_none_child(child)

        hierarchy = _merge_hierarchy_mask(hierarchy, masks)
        stack = _build_stack(hierarchy)
        while(stack):
            index = stack.pop()
            subtree = self._build_subtree(hierarchy, self.root, index, [])
            if(isinstance(subtree,list)):
                for st in subtree: self.root.add_child(st)
            else: self.root.add_child(subtree)

            _cut_off_none_child(self.root) 

    def _cut_off(self, value, node):
        for child in list(node.childs):
            if(child.mask.area < value):
                self.total_nodes -= 1 + len(child.childs)
                node.childs.remove(child)       
            else:
                self._cut_off(value, child)

    def cut_off(self, value):
        def _sum_nodes_area(node, total_area):
            try:
                total_area += node.mask.area
            except AttributeError as err:
                #print(err)
                pass
            for child in list(node.childs):
                total_area = _sum_nodes_area(child, total_area)

            return total_area


        value = value * _sum_nodes_area(self.root, 0) / self.total_nodes
        self._cut_off(value, self.root)

    def print_tree(self, simple=None):
        def _print_tree_node_simple(node, indent):
            print(indent, '[', node.name, ']')
            indent += '\t'
            for child in node.childs:
                _print_tree_node_simple(child, indent)

        def _print_tree_simple():
            print('[', self.root.name, ']')
            indent = '\t'
            for child in self.root.childs:
                _print_tree_node_simple(child, indent)

        def _print_tree_node_complete(node, indent):
            node.print_node(indent=indent)
            indent += '\t'
            for child in node.childs:
                _print_tree_node_complete(child, indent)

        def _print_tree_complete():
            self.root.print_node()
            indent = '\t'
            for child in self.root.childs:
                _print_tree_node_complete(child, indent)

        if simple:
            _print_tree_simple()
        else:
            _print_tree_complete()

    def _draw_tree(self, node, output):
        try:
            node.mask.draw('c', output, node.name)
        except AttributeError as err:
            #print(err)
            pass
        for child in list(node.childs):
            self._draw_tree(child, output)

    def draw_tree(self, output):
        self._draw_tree(self.root, output)

    def _get_tree_data(self, node, data):
        try:
            data.append(node)
        except AttributeError as err:
            #print(err)
            pass
        for child in list(node.childs):
            data = self._get_tree_data(child, data)
        
        return data

    def get_tree_data(self): #return list of nodes
        return self._get_tree_data(self.root, [])

    def _get_tree_masks(self, node, data):
        try:
            data.append(node.mask)
        except AttributeError as err:
            #print(err)
            pass
        for child in list(node.childs):
            data = self._get_tree_masks(child, data)
        
        return data

    def get_tree_masks(self): #return list of nodes
        return self._get_tree_masks(self.root, [])

    def _set_labels(self, node, labels):
        try:
            node.set_label(labels.pop(0))
        except AttributeError as err:
            #print(err)
            pass
        except IndexError as err:
            print("DEU UMA MERDA CABULOSA")
            print(err)
            exit(1)
        for child in list(node.childs):
            self._set_labels(child, labels)

    def set_labels(self, labels):
        '''
        root tem que ser vazio: [None] + labels
        '''
        labels = [None] + labels
        self._set_labels(self.root, labels)

    def _set_nodes_depth(self, node, depth, width):
        try:
            node.depth = depth
            if(depth not in width):
                width[depth] = 1
            else:
                width[depth] += 1
        except AttributeError as err:
            #print(err)
            pass
        except IndexError as err:
            print("DEU UMA MERDA CABULOSA")
            print(err)
            exit(1)
        
        for child in list(node.childs):
            _, width = self._set_nodes_depth(child, depth+1, width)

        node.neighbors = width[depth]
        return depth, width

    def set_nodes_depth(self):
        _, width = self._set_nodes_depth(self.root, 1, {})
        return width #dict depth : width

    def _get_apted(self, node, apted, depth, width):
        try:
            apted += '{'
            apted += str(node.label) if node.label != None else '-1'

            if(depth not in width):
                width[depth] = 1
            else:
                width[depth] += 1
        except AttributeError as err:
            #print(err)
            pass
        except IndexError as err:
            print("DEU UMA MERDA CABULOSA")
            print(err)
            exit(1)
        
        for child in list(node.childs):
            apted, _, width = self._get_apted(child, apted, depth+1, width)
            apted += '}'

        return apted, depth, width

    def set_apted(self):

        apted, _, width = self._get_apted(self.root, '', 1, {})
        apted += '}'

        return apted, width