def is_end_with(filename, extension):
    """
    判断文件名是否以指定后缀结尾
    参数:
    - filename: 字符串，文件名
    - extension: 字符串，需要检查的后缀，注意不需要点号
    返回:
    - 布尔值: 如果以指定后缀结尾返回True，否则返回False
    """
    if filename.lower().endswith(extension.lower()):
        return True
    else:
        return False
