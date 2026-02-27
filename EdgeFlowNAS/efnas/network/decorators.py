"""网络层装饰器定义。"""  # 定义模块用途

import tensorflow as tf  # 导入TensorFlow模块
from functools import wraps  # 导入装饰器工具


def count_and_scope(func):  # 定义计数+作用域装饰器
    """为层函数添加计数和命名作用域。"""  # 说明函数用途
    @wraps(func)  # 保留原函数元信息
    def wrapped(self, *args, **kwargs):  # 定义包装函数
        scope_name = f"{func.__name__}{self.curr_block}{self.suffix}"  # 生成作用域名称
        with tf.compat.v1.variable_scope(scope_name):  # 进入命名作用域
            self.curr_block += 1  # 递增当前块计数
            return func(self, *args, **kwargs)  # 调用原函数

    return wrapped  # 返回包装函数

