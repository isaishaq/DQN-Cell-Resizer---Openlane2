import re
from typing import Dict, List

def parse_timing_report(rpt_file: str) -> Dict:
    """Parse OpenSTA timing report into structured data."""
    
    paths = []
    current_path = None
    in_path_data = False
    
    with open(rpt_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect new path
        if line.startswith('Startpoint:'):
            if current_path:
                paths.append(current_path)
            current_path = {
                'startpoint': line.split(':')[1].strip(),
                'cells': []
            }
        
        elif line.startswith('Endpoint:') and current_path:
            current_path['endpoint'] = line.split(':')[1].strip()
        
        elif line.startswith('Path Group:') and current_path:
            current_path['path_group'] = line.split(':')[1].strip()
        
        elif line.startswith('Corner:') and current_path:
            current_path['corner'] = line.split(':')[1].strip()
        
        # Parse cell data in path
        elif 'Fanout' in line and 'Cap' in line:
            in_path_data = True
        
        elif in_path_data and '/' in line:
            # Parse cell instance line
            cell = parse_cell_line(line)
            if cell:
                current_path['cells'].append(cell)
        
        # Parse slack
        elif 'slack (VIOLATED)' in line or 'slack (MET)' in line:
            slack_match = re.search(r'(-?\d+\.\d+)\s+slack', line)
            if slack_match:
                current_path['slack'] = float(slack_match.group(1))
            in_path_data = False
        
        i += 1
    
    if current_path:
        paths.append(current_path)
    
    return {
        'paths': paths,
        'global_metrics': compute_global_metrics(paths)
    }

def parse_cell_line(line: str) -> Dict:
    """Parse individual cell line from timing path."""
    # Format: fanout cap slew delay time ^ cell_instance/pin (cell_type)
    parts = line.split()
    
    if len(parts) < 8:
        return None
    
    try:
        cell_info = {
            'fanout': int(parts[0]) if parts[0].isdigit() else 0,
            'cap': float(parts[1]),
            'slew': float(parts[2]),
            'delay': float(parts[3]),
            'time': float(parts[4]),
            'transition': parts[5],  # ^ or v
        }
        
        # Parse cell instance and type
        inst_part = ' '.join(parts[6:])
        if '/' in inst_part and '(' in inst_part:
            instance = inst_part.split('/')[0].strip()
            pin = inst_part.split('/')[1].split('(')[0].strip()
            cell_type = inst_part.split('(')[1].split(')')[0].strip()
            
            cell_info['instance_name'] = instance
            cell_info['pin'] = pin
            cell_info['cell_type'] = cell_type
            cell_info['drive_strength'] = extract_drive_strength(cell_type)
            cell_info['lib_cell'] = extract_lib_cell_type(cell_type)
        
        return cell_info
        
    except (ValueError, IndexError):
        return None

def extract_drive_strength(cell_type: str) -> int:
    """Extract drive strength from cell type name."""
    # e.g., sky130_fd_sc_hd__buf_4 -> 4
    match = re.search(r'_(\d+)$', cell_type)
    return int(match.group(1)) if match else 1

def extract_lib_cell_type(cell_type: str) -> str:
    """Extract base cell type."""
    # e.g., sky130_fd_sc_hd__buf_4 -> buf
    match = re.search(r'__([a-z0-9]+)_\d+', cell_type)
    return match.group(1) if match else cell_type

def compute_global_metrics(paths: List[Dict]) -> Dict:
    """Compute global timing metrics."""
    violations = [p for p in paths if p.get('slack', 0) < 0]
    
    return {
        'wns': min([p['slack'] for p in paths], default=0),
        'tns': sum([p['slack'] for p in violations]),
        'num_violations': len(violations),
        'num_paths': len(paths)
    }


if __name__ == "__main__":
    import json
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python parser.py <timing_report.rpt>")
        sys.exit(1)
    
    rpt_file = sys.argv[1]
    parsed_data = parse_timing_report(rpt_file)
    
    # Output structured data as JSON
    print(json.dumps(parsed_data, indent=4))