from typing import List, Optional

from .tank_exception import InvalidBasinFileException
from .basin_def import BasinDef, Subbasin, Reach, Junction, Sink


def extract_basin_file_data(content: str):
    sections = content.split("End:\n")
    data = []

    for section in sections:
        lines = section.strip().split("\n")
        section_data = {}
        current_key = None

        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                section_data[key.strip()] = value.strip()
                current_key = key.strip()
            elif line.strip() and current_key:
                section_data[current_key] += " " + line.strip()

        if section_data:
            data.append(section_data)

    return data


def extract_basin_defs_dict(basin_data: List[dict]):
    basin_defs_dict = {}

    for c in basin_data:
        basin_def: Optional[BasinDef] = None
        area = None
        downstream = None
        for k, v in c.items():
            if k == 'Subbasin':
                basin_def = Subbasin(v, 0.0)
            elif k == 'Reach':
                basin_def = Reach(v)
            elif k == 'Junction':
                basin_def = Junction(v)
            elif k == 'Sink':
                basin_def = Sink(v)
            elif k == 'Area':
                area = float(v)
            elif k == 'Downstream':
                downstream = v

        if basin_def is not None:
            if area is not None and isinstance(basin_def, Subbasin):
                basin_def.area = area
            basin_defs_dict[basin_def.name] = basin_def, downstream

    return basin_defs_dict


def build_root_node(basin_defs: List[BasinDef]):
    root_node = []
    for basin_def in basin_defs:
        ds = basin_def.downstream
        if ds is None:
            if not root_node:
                root_node = [basin_def]
            else:
                root_node.append(basin_def)
        else:
            if not ds.upstream:
                ds.upstream = [basin_def]
            else:
                ds.upstream.append(basin_def)

    return root_node


def build_basin_def_and_root_node(basin_file: str):
    try:
        basin_data = extract_basin_file_data(open(basin_file, 'r').read())
        basin_defs_dict = extract_basin_defs_dict(basin_data)

        for basin_def_name, basin_def in basin_defs_dict.items():
            basin_def, downstream = basin_def
            if downstream is not None:
                basin_def.downstream = basin_defs_dict[downstream][0]

        basin_defs = [basin_def[0] for basin_def in basin_defs_dict.values()]
        root_node = build_root_node(basin_defs)

    except Exception as _:
        print(_)
        raise InvalidBasinFileException()

    return basin_defs, root_node
