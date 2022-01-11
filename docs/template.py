import sys
import re
import os

class Template():
    def __init__(self, file, indent=0):
        self.template_dir = os.path.dirname(file)
        with open(file) as f:
            self.text = f.read()
        self.var_pat = "{{.*?}}"
        self.tag_pat = "{%.*?%}"
        self.tag_end_pat = "{%\s*?" "end(if|for)" "\s*?%}"
        self.tags_without_block = ["include"]
        self.indent = indent

    def render(self, **kwargs):
        content = self._render_core(self.text, kwargs)

        # add indent
        content, tmp = [], content
        for line in tmp.rstrip().split("\n"):
            content.append(" "*self.indent + line + "\n")

        return "".join(content)

    def _render_core(self, text, table, local_table=None):
        if local_table is None:
            local_table = {}
        content = []
        tag_idx = []
        while True:
            pre, statement, text = self._search(f"{self.var_pat}|{self.tag_pat}", text)
            content.append(pre)
            if statement is None:
                break

            if re.match(f"{self.var_pat}", statement):
                content.append(self._render_var(statement, table, local_table))
            elif re.match(f"{self.tag_pat}", statement):
                indent = len(content[-1]) - len(content[-1].rstrip(" "))
                content[-1] = content[-1].rstrip(" ")
                text = text.lstrip(" ").lstrip("\n")

                tag, expr = self._rm_outer(statement).split(maxsplit=1)
                if tag != "include":
                    # find corresponding tag end
                    block, text = self._get_tag_block(text)
                    text = text.lstrip(" ").lstrip("\n")
                    block = block.rstrip(" ")


                if tag == "if":
                    if self._eval(expr, table, local_table):
                        content.append(self._render_core(block, table, local_table))
                elif tag == "for":
                    lv, rv = map(lambda s: s.strip(), expr.split("in"))
                    for v in self._eval(rv, table, local_table):
                        local_table[lv] = v
                        content.append(self._render_core(block, table, local_table))
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
                            lv, rv = match.group().split("=")
                            kwargs[lv] = self._eval(rv, table, local_table)

                    template = Template(os.path.join(self.template_dir, eval(file)), indent)
                    content.append(template.render(**{**table, **kwargs}))
                else:
                    raise Exception(f"Unsupported statement: {statement}")

        return "".join(content)

    def _search(self, pat, text):
        match = re.search(pat, text)
        if match is None:
            return text, None, text
        pre = text[:match.start()]
        statement = match.group()
        post = text[match.end():]
        return pre, statement, post

    def _get_tag_block(self, text):
        block = []
        count = 0
        while True:
            pre, statement, text = self._search(f"{self.tag_pat}", text)
            block.append(pre)

            if re.match(f"{self.tag_end_pat}", statement):
                count -= 1
            else:
                has_block = True
                for tag in self.tags_without_block:
                    if re.match("{%\s*?" f"{tag}" ".*?%}", statement):
                        has_block = False
                if has_block:
                    count += 1

            if count < 0:
                return "".join(block), text
            else:
                block.append(statement)

    def _rm_outer(self, statement):
        statement = statement.strip()
        if re.match(f"{self.var_pat}|{self.tag_pat}", statement):
            return statement[2:-2].strip()
        else:
            return statement.strip()

    def _update_locals(self, loc, *dicts):
        for d in dicts:
            loc.update(d)

    def _eval(_self, _expr, _table, _local_table):
        locals().update(_table)
        locals().update(_local_table)
        return eval(_expr)

    def _render_var(self, string, table, local_table):
        #self._update_locals(locals(), table, local_table)
        name = self._rm_outer(string)
        #return str(table[name] if name in table else local_table[name])
        try:
            return str(self._eval(name, table, local_table))
        except NameError:
            return string

if __name__ == '__main__':
    t = Template(sys.argv[1])
    print(t.text)
    print(t.render(haha="haha_text", a=5, b=6, c=1))
