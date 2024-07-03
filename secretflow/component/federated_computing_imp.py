import re
import json
import jax
import pandas as pd


class ExpressionToSf():
    def __init__(self):
        self.precedence = {
            '+': 1, '-': 1, '*': 2, '/': 2, '>': 0, '<': 0, '>=': 0, '<=': 0, '!=': 0, '&': -1, '|': -2, '!': 3
        }
        self.pattern = re.compile(r'([a-zA-Z_\u4e00-\u9fa5]+\w*|\d+|[()+\-*/<>&|!]=?)')  # 定义匹配算术表达式的正则表达式模式
        self.op = ['+', '-', '*', '/', '>', '<', '>=', '<=', '&', '|', '!']
        self.columns = []

    @staticmethod
    def contains_valid_chars(input_str):
        return bool(re.match(r'^[\w\u4e00-\u9fa5]+$', input_str))

    def parse_expression_to_json(self, expression):
        # 切分表达式为token
        tokens = self.pattern.findall(expression)
        # 将token列表转换为逆波兰表达式（后缀表达式）
        output, operators, stack = [], [], []
        for token in tokens:
            if self.contains_valid_chars(token):  # 如果是变量名或数字
                self.columns.append(token)
                output.append(token)
            elif token in self.precedence:  # 如果是操作符
                while (operators and operators[-1] != '(' and
                       self.precedence[operators[-1]] >= self.precedence[token]):
                    output.append(operators.pop())
                operators.append(token)
            elif token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()  # 弹出左括号
        while operators:
            output.append(operators.pop())

        for token in output:
            if token in self.op:
                right = stack.pop()
                left = stack.pop()
                rule_name = f"rule{len(stack) + 1}"
                stack.append({
                    "op": "add" if token == '+' else "sub" if token == '-' else "mul" if token == '*' else "div"
                    if token == "/" else "less" if token == "<" else "greater" if token == ">" else "equal"
                    if token == "=" else "less_equal" if token == "<=" else "greater_equal"
                    if token == ">=" else "not_equal" if token == "!=" else "and"
                    if token == "&" else "or" if token == "|" else "not",
                    "args": [left, right]
                })
            else:
                stack.append(token)
        return json.dumps(stack[0], indent=2)

    def json_to_function(self, json_expr, v_df, spu):
        # 解析 JSON
        parsed = json.loads(json_expr)
        print("-----------", parsed)

        # 递归构建目标字符串
        def build_expression(obj):
            if isinstance(obj, dict):
                if obj["op"] == "add":
                    return spu(add_calculation)(build_expression(obj['args'][0]), build_expression(obj['args'][1]))
                elif obj["op"] == "sub":
                    return spu(sub_calculation)(build_expression(obj['args'][0]), build_expression(obj['args'][1]))
                elif obj["op"] == "mul":
                    return spu(mul_calculation)(build_expression(obj['args'][0]), build_expression(obj['args'][1]))
                elif obj["op"] == "div":
                    return spu(div_calculation)(build_expression(obj['args'][0]), build_expression(obj['args'][1]))
                elif obj["op"] == "less":
                    return spu(less_calculation)(build_expression(obj['args'][0]), build_expression(obj['args'][1]))
                elif obj["op"] == "greater":
                    return spu(greater_calculation)(build_expression(obj['args'][0]), build_expression(obj['args'][1]))
                elif obj["op"] == "equal":
                    return spu(equal_calculation)(build_expression(obj['args'][0]), build_expression(obj['args'][1]))
                elif obj["op"] == "less_equal":
                    return spu(less_equal_calculation)(build_expression(obj['args'][0]),
                                                       build_expression(obj['args'][1]))
                elif obj["op"] == "greater_equal":
                    return spu(greater_equal_calculation)(build_expression(obj['args'][0]),
                                                          build_expression(obj['args'][1]))
                elif obj["op"] == "not_equal":
                    return spu(not_equal_calculation)(build_expression(obj['args'][0]),
                                                      build_expression(obj['args'][1]))
                elif obj["op"] == "and":
                    return spu(logical_and)(build_expression(obj['args'][0]), build_expression(obj['args'][1]))
                elif obj["op"] == "or":
                    return spu(logical_or)(build_expression(obj['args'][0]), build_expression(obj['args'][1]))
                elif obj["op"] == "not":
                    return spu(logical_not)(build_expression(obj['args'][0]), build_expression(obj['args'][1]))
            elif isinstance(obj, list):
                return ", ".join([build_expression(item) for item in obj])
            else:
                return v_df.get_flat_column(obj)  # 可能是变量名或者数字

        # 构建最终的表达式
        result = build_expression(parsed)
        print("------------------------------", result)
        return result

    def formula_to_function(self, expression, v_df, spu):
        result = self.parse_expression_to_json(expression)
        print("result", result)

        return self.json_to_function(result, v_df, spu)


def read_data_with_pandas(path):
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")
    return df


def add_calculation(x, y):
    return jax.numpy.add(x, y)


def sub_calculation(x, y):
    return jax.numpy.subtract(x, y)


def mul_calculation(x, y):
    return jax.numpy.multiply(x, y)


def div_calculation(x, y):
    return jax.numpy.divide(x, y)


def greater_calculation(x, y):
    return jax.numpy.greater(x, y)


def greater_equal_calculation(x, y):
    return jax.numpy.greater_equal(x, y)


def less_calculation(x, y):
    return jax.numpy.less(x, y)


def less_equal_calculation(x, y):
    return jax.numpy.less_equal(x, y)


def equal_calculation(x, y):
    return jax.numpy.equal(x, y)


def not_equal_calculation(x, y):
    return jax.numpy.not_equal(x, y)


def logical_and(x, y):
    return jax.numpy.logical_and(x, y)


def logical_or(x, y):
    return jax.numpy.logical_or(x, y)


def logical_not(x, y):
    return jax.numpy.logical_not(x)


if __name__ == '__main__':
    sf_op = ExpressionToSf()
    # 测试
    expression = "(x_1-x_2 + x_11/x_6 *(x_13+x_2)) + x_1"
    # res = sf_op.formula_to_function(expression)
