"""
Timing Report Parser for OpenSTA
=================================

Parses OpenSTA timing reports into structured data for RL agent.
"""

import re
from typing import Dict, List, Optional
from pathlib import Path


class TimingReportParser:
    """Parse OpenSTA timing reports (.rpt files)."""
    
    def __init__(self):
        self.paths = []
        self.current_path = None
        
    def parse_file(self, report_path: str) -> Dict:
        """
        Parse timing report file.
        
        Args:
            report_path: Path to .rpt file
            
        Returns:
            Dictionary with paths and global metrics
        """
        report_path = Path(report_path)
        if not report_path.exists():
            raise FileNotFoundError(f"Report not found: {report_path}")
        
        self.paths = []
        self.current_path = None
        
        with open(report_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        in_path_section = False
        
        while i < len(lines):
            line = lines[i]
            
            # Detect new path
            if line.startswith('Startpoint:'):
                if self.current_path:
                    self.paths.append(self.current_path)
                
                self.current_path = {
                    'startpoint': self._extract_value(line),
                    'cells': [],
                    'endpoint': '',
                    'path_group': '',
                    'corner': '',
                    'slack': 0.0
                }
            
            elif line.startswith('Endpoint:') and self.current_path:
                self.current_path['endpoint'] = self._extract_value(line)
            
            elif line.startswith('Path Group:') and self.current_path:
                self.current_path['path_group'] = self._extract_value(line)
            
            elif line.startswith('Corner:') and self.current_path:
                self.current_path['corner'] = self._extract_value(line)
            
            # Detect start of path data
            elif 'Fanout' in line and 'Cap' in line and 'Slew' in line:
                in_path_section = True
            
            # Parse cell lines
            elif in_path_section and '/' in line and '(' in line:
                cell = self._parse_cell_line(line)
                if cell and self.current_path:
                    self.current_path['cells'].append(cell)
            
            # Parse slack
            elif 'slack' in line.lower():
                slack_match = re.search(r'(-?\d+\.\d+)\s+slack', line)
                if slack_match and self.current_path:
                    self.current_path['slack'] = float(slack_match.group(1))
                in_path_section = False
            
            # Parse data arrival/required time
            elif 'data arrival time' in line.lower():
                time_match = re.search(r'(-?\d+\.\d+)\s+data arrival time', line)
                if time_match and self.current_path:
                    self.current_path['data_arrival_time'] = float(time_match.group(1))
            
            elif 'data required time' in line.lower():
                time_match = re.search(r'(-?\d+\.\d+)\s+data required time', line)
                if time_match and self.current_path:
                    self.current_path['data_required_time'] = float(time_match.group(1))
            
            i += 1
        
        # Add last path
        if self.current_path:
            self.paths.append(self.current_path)
        
        # Compute global metrics
        result = {
            'paths': self.paths,
            'global_metrics': self._compute_global_metrics()
        }
        
        return result
    
    def _extract_value(self, line: str) -> str:
        """Extract value after colon."""
        parts = line.split(':', 1)
        if len(parts) == 2:
            return parts[1].strip()
        return ''
    
    def _parse_cell_line(self, line: str) -> Optional[Dict]:
        """
        Parse individual cell line from timing path.
        
        Format:
        fanout cap slew delay time transition instance/pin (cell_type)
        """
        # Remove leading whitespace and split
        parts = line.split()
        
        if len(parts) < 7:
            return None
        
        try:
            # Find the instance/pin part (contains '/')
            inst_pin_idx = None
            for i, part in enumerate(parts):
                if '/' in part:
                    inst_pin_idx = i
                    break
            
            if inst_pin_idx is None:
                return None
            
            # Parse numeric values (before instance name)
            fanout = int(parts[0]) if parts[0].isdigit() else 0
            cap = float(parts[1])
            slew = float(parts[2])
            delay = float(parts[3])
            time = float(parts[4])
            transition = parts[5]  # ^ or v
            
            # Parse instance/pin and cell type
            inst_pin_str = parts[inst_pin_idx]
            remaining = ' '.join(parts[inst_pin_idx:])
            
            # Extract instance name and pin
            if '/' in inst_pin_str:
                instance = inst_pin_str.split('/')[0]
                pin_and_rest = inst_pin_str.split('/')[1]
                
                # Extract pin (before parenthesis)
                if '(' in pin_and_rest:
                    pin = pin_and_rest.split('(')[0].strip()
                else:
                    pin = pin_and_rest.strip()
            else:
                return None
            
            # Extract cell type (inside parentheses)
            cell_type = ''
            if '(' in remaining and ')' in remaining:
                cell_type = remaining.split('(')[1].split(')')[0].strip()
            
            # Extract drive strength and base cell type
            drive_strength = self._extract_drive_strength(cell_type)
            base_cell_type = self._extract_base_cell_type(cell_type)
            
            return {
                'fanout': fanout,
                'cap': cap,
                'slew': slew,
                'delay': delay,
                'time': time,
                'transition': transition,
                'instance_name': instance,
                'pin': pin,
                'cell_type': cell_type,
                'drive_strength': drive_strength,
                'base_cell_type': base_cell_type
            }
        
        except (ValueError, IndexError) as e:
            return None
    
    def _extract_drive_strength(self, cell_type: str) -> int:
        """Extract drive strength from cell type name."""
        # e.g., sky130_fd_sc_hd__buf_4 -> 4
        match = re.search(r'_(\d+)$', cell_type)
        return int(match.group(1)) if match else 1
    
    def _extract_base_cell_type(self, cell_type: str) -> str:
        """Extract base cell type without library prefix and drive strength."""
        # e.g., sky130_fd_sc_hd__buf_4 -> buf
        match = re.search(r'__([a-z0-9]+)_\d+', cell_type)
        if match:
            return match.group(1)
        # Handle cells without drive strength
        match = re.search(r'__([a-z0-9]+)', cell_type)
        return match.group(1) if match else cell_type
    
    def _compute_global_metrics(self) -> Dict:
        """Compute global timing metrics."""
        violations = [p for p in self.paths if p.get('slack', 0) < 0]
        
        if not self.paths:
            return {
                'wns': 0.0,
                'tns': 0.0,
                'num_violations': 0,
                'num_paths': 0
            }
        
        wns = min([p.get('slack', 0) for p in self.paths])
        tns = sum([p.get('slack', 0) for p in violations])
        
        return {
            'wns': wns,
            'tns': tns,
            'num_violations': len(violations),
            'num_paths': len(self.paths)
        }
    
    def get_critical_cells(self, top_n_paths: int = 10, 
                          min_delay: float = 0.1) -> List[Dict]:
        """
        Get most critical cells across top N paths.
        
        Args:
            top_n_paths: Number of worst paths to analyze
            min_delay: Minimum delay to consider cell as critical
            
        Returns:
            List of critical cells with metadata
        """
        # Sort paths by slack
        sorted_paths = sorted(self.paths, key=lambda p: p.get('slack', 0))
        critical_paths = sorted_paths[:top_n_paths]
        
        # Aggregate cell criticality
        cell_info = {}  # instance_name -> cell info
        
        for path in critical_paths:
            path_slack = path.get('slack', 0)
            cells = path.get('cells', [])
            
            for i, cell in enumerate(cells):
                instance = cell.get('instance_name', '')
                if not instance:
                    continue
                
                # Skip low-delay cells
                if cell.get('delay', 0) < min_delay:
                    continue
                
                # Calculate criticality score
                position_weight = (i + 1) / len(cells) if cells else 0
                delay = cell.get('delay', 0)
                slack_weight = abs(path_slack) if path_slack < 0 else 0
                criticality = delay * position_weight * (1 + slack_weight)
                
                if instance not in cell_info:
                    cell_info[instance] = {
                        'instance_name': instance,
                        'cell_type': cell.get('cell_type', ''),
                        'base_cell_type': cell.get('base_cell_type', ''),
                        'drive_strength': cell.get('drive_strength', 1),
                        'fanout': cell.get('fanout', 0),
                        'avg_delay': cell.get('delay', 0),
                        'criticality': criticality,
                        'path_count': 1
                    }
                else:
                    # Cell appears in multiple paths
                    info = cell_info[instance]
                    info['criticality'] += criticality
                    info['path_count'] += 1
                    info['avg_delay'] = (info['avg_delay'] * (info['path_count'] - 1) + 
                                        cell.get('delay', 0)) / info['path_count']
        
        # Sort by criticality
        critical_cells = sorted(cell_info.values(), 
                               key=lambda x: x['criticality'], 
                               reverse=True)
        
        return critical_cells


def parse_timing_report(report_path: str) -> Dict:
    """
    Convenience function to parse timing report.
    
    Args:
        report_path: Path to timing report file
        
    Returns:
        Parsed timing data dictionary
    """
    parser = TimingReportParser()
    return parser.parse_file(report_path)


# ============================================================================
# Main - For testing
# ============================================================================

if __name__ == '__main__':
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python timing_parser.py <report_file.rpt> [output.json]")
        sys.exit(1)
    
    report_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Parsing timing report: {report_file}")
    
    try:
        data = parse_timing_report(report_file)
        
        print(f"\nGlobal Metrics:")
        print(f"  WNS: {data['global_metrics']['wns']:.3f} ns")
        print(f"  TNS: {data['global_metrics']['tns']:.3f} ns")
        print(f"  Violations: {data['global_metrics']['num_violations']}")
        print(f"  Total Paths: {data['global_metrics']['num_paths']}")
        
        # Get critical cells
        parser = TimingReportParser()
        parser.paths = data['paths']
        critical_cells = parser.get_critical_cells(top_n_paths=10)
        
        print(f"\nTop 10 Critical Cells:")
        for i, cell in enumerate(critical_cells[:10]):
            print(f"  {i+1}. {cell['instance_name']}: "
                  f"{cell['cell_type']} "
                  f"(criticality={cell['criticality']:.3f}, "
                  f"paths={cell['path_count']})")
        
        # Save to JSON if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\nSaved parsed data to: {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
