import streamlit as st
from sympy import (sin, cos, tan, cot, sec, csc,
                   asin, acos, atan, acot, asec, acsc,
                   sinh, cosh, tanh, coth, sech, csch,
                   asinh, acosh, atanh, acoth, asech, acsch,
                   exp, log, Abs, factorial, sqrt, root, E,
                   pi, symbols, latex, lambdify)
from sympy.parsing.sympy_parser import (standard_transformations,
                                        implicit_multiplication_application,
                                        convert_xor, split_symbols,
                                        function_exponentiation, TokenError,
                                        parse_expr)
import re
import plotly.graph_objects as go
import numpy as np

x = symbols('x', real=True)

st.set_page_config(
    page_title="MathGraphs",
    page_icon="math_icon.png",
    layout="wide"
)

transformations = (
    standard_transformations +
    (implicit_multiplication_application, convert_xor,
     split_symbols, function_exponentiation)
)


def is_expression_too_complex(expr_str, max_length=100, max_depth=4):
    if len(expr_str) > max_length:
        return True

    depth = 0
    max_depth_found = 0
    for char in expr_str:
        if char == '(':
            depth += 1
            max_depth_found = max(max_depth_found, depth)
        elif char == ')':
            depth -= 1
        if depth < 0 or max_depth_found > max_depth:
            return True

    dangerous_patterns = [
        r'factorial\s*\(.*factorial',
        r'!\s*[0-9]{4,}',
        r'x\s*\*\*\s*[0-9]{3,}',
        r'repeat|while|for|import'
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, expr_str):
            return True

    return False


def fix_math_functions(text):
    text = re.sub(r'(\d+)!', r'factorial(\1)', text)

    text = text.replace('^', '**')

    text = re.sub(
        r'(\d*)(sin|cos|tan|tg|cot|ctg|sec|csc|'
        r'asin|acos|atan|acot|asec|acsc|'
        r'sinh|cosh|tanh|coth|sech|csch|'
        r'asinh|acosh|atanh|acoth|asech|acsch|'
        r'exp|log|ln|abs|sqrt)([a-zA-Z0-9]+)',
        lambda m: f"{m.group(1)}*{m.group(2).lower()}({m.group(3)})" if m.group(1)
        else f"{m.group(2).lower()}({m.group(3)})",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r'(\d+)([a-zA-Z(])', r'\1*\2', text)
    text = re.sub(r'([a-zA-Z)])(\d+)', r'\1*\2', text)

    return text


def parse_function(func_str):
    if not func_str.strip():
        return None, "Введите функцию, поле не может быть пустым"

    if is_expression_too_complex(func_str):
        return None, "Слишком сложное выражение. Упростите его"

    error_messages = {
        f"name 'x' is not defined": f"Используйте 'x' как переменную",
        "invalid syntax": "Ошибка в синтаксисе выражения",
        "unexpected EOF": "Незавершённое выражение",
        "could not convert string to float": "Недопустимые символы в выражении",
        "TokenError": "Недопустимый символ в выражении"
    }

    local_dict = {
        'x': x,
        'sin': sin, 'cos': cos, 'tan': tan,
        'tg': tan, 'cot': cot, 'ctg': cot,
        'sec': sec, 'csc': csc,
        'asin': asin, 'arcsin': asin,
        'acos': acos, 'arccos': acos,
        'atan': atan, 'arctan': atan, 'atg': atan, 'arctg': atan,
        'acot': acot, 'arccot': acot, 'actg': acot, 'arcctg': acot,
        'asec': asec, 'arcsec': asec,
        'acsc': acsc, 'arccsc': acsc,
        'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
        'coth': coth, 'sech': sech, 'csch': csch,
        'asinh': asinh, 'arcsinh': asinh,
        'acosh': acosh, 'arccosh': acosh,
        'atanh': atanh, 'arctanh': atanh,
        'acoth': acoth, 'arccoth': acoth,
        'asech': asech, 'arcsech': asech,
        'acsch': acsch, 'arccsch': acsch,
        'exp': exp, 'log': log, 'ln': log,
        'sqrt': sqrt, 'root': root,
        'abs': Abs, 'Abs': Abs,
        'factorial': factorial, '!': factorial,
        'pi': pi, 'e': E, 'E': E
    }

    try:
        p_expr = parse_expr(func_str, transformations=transformations, local_dict=local_dict)

        undefined = [str(s) for s in p_expr.free_symbols if str(s) not in local_dict]
        if undefined:
            return None, f"Неопределённые символы: {', '.join(map(str, undefined))}"

        return p_expr, None

    except TokenError as e:
        return None, f"Ошибка в позиции {e.args[1]}: неправильный символ"
    except SyntaxError:
        return None, "Ошибка синтаксиса: проверьте скобки и операторы"
    except Exception as e:
        for pattern, msg in error_messages.items():
            if pattern in str(e):
                return None, msg
        return None, f"Ошибка: {str(e)}"


def generate_x_values(x_max=1e6, num_points=5000):
    x_linear = np.linspace(-100, 100, num_points // 2)

    x_log_positive = np.logspace(2, np.log10(x_max), num=num_points // 4)
    x_log_negative = -np.logspace(2, np.log10(x_max), num=num_points // 4)

    x_vals = np.unique(np.concatenate([x_log_negative, x_linear, x_log_positive]))
    return x_vals


def plot_with_plotly(expr, x_max=1e6):
    try:
        x_vals = generate_x_values(x_max)
        f = lambdify(x, expr, modules=['numpy'])
        y_vals = f(x_vals)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            line=dict(width=1.5, color='blue'),
            name='f(x)'
        ))

        fig.update_xaxes(
            type='linear',
            range=[-100, 100],
            autorange=True,
            constrain="domain"
        )

        fig.update_layout(
            title=f'График функции:',
            xaxis_title='x',
            yaxis_title='f(x)',
            hovermode='x unified',
            showlegend=False,
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Линейный масштаб",
                        "method": "relayout",
                        "args": [{"xaxis.type": "linear"}]
                    },
                    {
                        "label": "Логарифмический масштаб",
                        "method": "relayout",
                        "args": [{"xaxis.type": "log"}]
                    }
                ]
            }]
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка построения графика: {str(e)}")


st.title('Построение графиков функций')

if 'function' not in st.session_state:
    st.session_state.function = "sin(x)"
if "last_valid_function" not in st.session_state:
    st.session_state.last_valid_function = "sin(x)"

st.header('Текущая функция')
fixed_function = fix_math_functions(st.session_state.last_valid_function)
expr, _ = parse_function(fixed_function)
st.latex(f'f(x) = {latex(expr)}')

new_function = st.text_input(
    "Введите функцию",
    value=st.session_state.function,
    help="В качестве переменной стоит использовать 'x'"
)

x_max = st.slider(
    "Максимальное значение |x|",
    min_value=100,
    max_value=10_000,
    value=100,
    step=100,
    format="%d"
)

if new_function != st.session_state.function:
    fixed_function = fix_math_functions(new_function)
    expr, error = parse_function(fixed_function)

    if error:
        st.error(error)
    else:
        st.session_state.function = new_function
        st.session_state.last_valid_function = new_function
        st.rerun()

if expr:
    plot_with_plotly(expr, x_max=x_max)

st.markdown("---")
