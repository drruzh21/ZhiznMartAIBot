import os

def get_project_file_path(filename: str) -> str:
    """
    Функция для получения полного пути к файлу в проекте, относительно корневой директории.

    Args:
    filename (str): Имя файла, к которому нужно сформировать путь.

    Returns:
    str: Полный путь к файлу.
    """
    # Получаем абсолютный путь к корневой директории проекта
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Формируем полный путь к файлу, используя имя файла
    file_path = os.path.join(project_root, filename)

    # Возвращаем полный путь
    return file_path
