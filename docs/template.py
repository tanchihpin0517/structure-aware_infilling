import sys
import re
import os

class Node:
    def __init__(self, parent=None, content=None, indent=0):
        self.parent = parent
        self.content = content
        self.children = []
        self.indent = indent

    def __repr__(self):
        return "%r" % self.content

    def __str__(self):
        return "%r" % self.content

    def __iter__(self):
        return self.children.__iter__()

    def __len__(self):
        return len(self.children)

    def __getitem__(self, key):
        return self.children[key]

    def add_child(self, node):
        self.children.append(node)

    def preorder(self):
        s = [str(self)]
        for child in self.children:
            s.append(child.preorder())
            s.append('\n')
        return "".join(s)

class Template:
    def __init__(self, file, indent=0):
        self.template_dir = os.path.dirname(file)
        with open(file) as f:
            self.text = f.read()
        self.var_pat = "{{.*?}}"
        self.tag_pat = "{%.*?%}"
        self.tag_end_pat = "{%\s*?" "end(if|for)" "\s*?%}"
        self.tag_single_pat = "|".join(map(self._get_tag_pat, ["include", "elif", "else", "with"]))
        self.tree = self._parse_tree(self.text)

    def _get_tag_pat(self, tag):
        return "{%.*?" + tag + ".*?%}"

    def _parse_tree(self, string):
        root = cur_node = Node()
        for line in string.split("\n"):
            # record indent
            indent = len(line) - len(line.strip())

            # strip if the line contains a tag only
            prev, statement, post = self._search(self.tag_pat, line.strip())
            if statement is not None and (len(prev) == 0 and len(post) == 0):
                text = line.strip()
            else:
                text = line + "\n"

            # build tree
            while True:
                prev, statement, text = self._search(f"{self.var_pat}|{self.tag_pat}", text)
                cur_node.add_child(Node(parent=cur_node, content=prev))
                if statement is None:
                    break

                if re.match(self.tag_end_pat, statement):
                    cur_node = cur_node.parent
                else:
                    node = Node(parent=cur_node, content=statement, indent=indent)
                    cur_node.add_child(node)
                    if re.match(self.tag_pat, statement) and not re.match(self.tag_single_pat, statement):
                        cur_node = node
        return root

    def render(self, var_table, indent=0):
        content = []
        local_table = {}
        for child in self.tree:
            content.append(self._render_core(child, var_table, local_table))
        content = "".join(content)

        # add indent
        content, tmp = [], content
        for line in tmp.rstrip().split("\n"):
            content.append(" "*indent + line + "\n")

        return "".join(content)

    def _render_core(self, node, table, local_table=None):
        if local_table is None:
            local_table = {}

        content = []

        if re.match(f"{self.var_pat}", node.content):
            content.append(self._render_var(node.content, table, local_table))
        elif re.match(f"{self.tag_pat}", node.content):
            tag, expr = self._rm_outer(node.content).split(maxsplit=1)

            if tag == "if":
                """
                if ... elif ... else ...
                """
                # test condition
                if self._eval(expr, table, local_table):
                    for child in node:
                        if re.match(self._get_tag_pat("else|elif"), child.content):
                            break
                        content.append(self._render_core(child, table, local_table))
                else:
                    for i, child in enumerate(node):
                        #print(child)
                        if re.match(self._get_tag_pat("elif"), child.content):
                            tag, expr = self._rm_outer(child.content).split(maxsplit=1)
                            # test condition
                            if self._eval(expr, table, local_table):
                                # start at next child
                                for child in node[i+1:]:
                                    # meet the end of block
                                    if re.match(self._get_tag_pat("else|elif"), child.content):
                                        break
                                    content.append(self._render_core(child, table, local_table))
                                break
                        elif re.match(self._get_tag_pat("else"), child.content):
                            # render without conditions
                            for child in node[i+1:]:
                                # meet the end of block
                                if re.match(self._get_tag_pat("else|elif"), child.content):
                                    break
                                content.append(self._render_core(child, table, local_table))

            elif tag == "for":
                """
                for v in iterable
                for v... in (iterable of iterable)
                """
                lexpr, rexpr = map(lambda s: s.strip(), expr.split("in"))
                var_names = lexpr.split(",")
                vs = [0] * len(var_names)
                if len(var_names) == 1: # normal form
                    for vs[0] in self._eval(rexpr, table, local_table):
                        local_table[var_names[0]] = vs[0]
                        content.append(self._render_children(node, table, local_table))
                else: # for loop unpacking
                    for vs in self._eval(rexpr, table, local_table):
                        for i in range(len(vs)):
                            local_table[var_names[i]] = vs[i]
                        content.append(self._render_children(node, table, local_table))

            elif tag == "include":
                tokens = expr.split("with")
                if len(tokens) == 1:
                    file = expr
                else:
                    file, assignments = tokens
                    kwargs = {}
                    while True:
                        match = re.search("\S*\s*=\s*\S*", assignments)
                        if match is None:
                            break
                        assignments = assignments[match.end():]
                        name, value = match.group().split("=")
                        kwargs[name] = self._eval(value, table, local_table)

                template = Template(os.path.join(self.template_dir, eval(file)))
                content.append(template.render({**table, **kwargs}, node.indent))

            elif tag == "with":
                assignments = expr
                while True:
                    match = re.search("\S*\s*=\s*\S*", assignments)
                    if match is None:
                        break
                    assignments = assignments[match.end():]
                    name, value = match.group().split("=")
                    local_table[name] = self._eval(value, table, local_table)

            else:
                raise Exception(f"Unsupported statement: {node.content}")
        else:
            content.append(node.content)

        return "".join(content)

    def _render_children(self, node, table, local_table):
        content = []
        for child in node:
            content.append(self._render_core(child, table, local_table))
        return "".join(content)

    def _search(self, pat, text):
        match = re.search(pat, text)
        if match is None:
            return text, None, text
        pre = text[:match.start()]
        statement = match.group()
        post = text[match.end():]
        return pre, statement, post

    def _rm_outer(self, statement):
        statement = statement.strip()
        if re.match(f"{self.var_pat}|{self.tag_pat}", statement):
            return statement[2:-2].strip()
        else:
            return statement.strip()

    def _eval(_self, _expr, _table, _local_table):
        locals().update(_table)
        locals().update(_local_table)
        return eval(_expr)

    def _render_var(self, string, table, local_table):
        name = self._rm_outer(string)
        try:
            return str(self._eval(name, table, local_table))
        except NameError:
            print("Variable not find: %s" % name)
            return string

if __name__ == '__main__':
    t = Template(sys.argv[1])
    print("==============================")
    print(t.text)
    print("==============================")
    table = {"haha":"haha_text", 'a':5, 'b':6, 'c':1}
    content = t.render(table)
    print(content)
