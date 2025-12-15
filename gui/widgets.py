# widgets.py

# IMPORTS

# GUI imports
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QGroupBox, QGridLayout, QTabWidget,
    QTextEdit, QSpinBox, QDoubleSpinBox, QSplitter, QLineEdit, QSizePolicy,
    QScrollArea, QCheckBox
    )
from PySide6.QtCore import (
    Qt, QThread, Signal
    )
from PySide6.QtGui import QFont

# 3D Visualization imports
import pyvista as pv
from pyvistaqt import QtInteractor

# System imports
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]  # parent_directory
src_path = root / "src" / "python"
sys.path.append(str(src_path))
import fk # pyright: ignore[reportMissingImports]
import fkp # pyright: ignore[reportMissingImports]
import spline # pyright: ignore[reportMissingImports]
from materials_library import MaterialLibrary # pyright: ignore[reportMissingImports]
import checks # pyright: ignore[reportMissingImports]
import ik # pyright: ignore[reportMissingImports]
import time
from functools import wraps

# CUDA imports
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes

# Other imports
import math

import time
import numpy as np  # Already imported, but ensure it's there

# LOAD CUDA SHARED LIBRARY
lib = ctypes.CDLL(str(root / "lib/fk.so"))
lib.generate_spline_points.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")
]
lib.generate_spline_points.restype = ctypes.c_int

lib.initialize_gpu_context.argtypes = [
    ctypes.c_int,                                      # num_segments
    ctypes.c_int,                                      # resolution
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),   # length
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")    # T_cumulative
]
lib.initialize_gpu_context.restype = ctypes.c_int

lib.update_spline_fast.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),   # kappa
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),   # theta
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),   # phi
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")    # output
]
lib.update_spline_fast.restype = ctypes.c_int

lib.update_transforms.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")    # T_cumulative
]
lib.update_transforms.restype = ctypes.c_int
lib.check_gpu_status.argtypes = []
lib.check_gpu_status.restype = ctypes.c_int
lib.destroy_gpu_context.argtypes = []
lib.destroy_gpu_context.restype = ctypes.c_int
lib.reset_gpu_context.argtypes = []
lib.reset_gpu_context.restype = ctypes.c_int

# ADD THIS TEST CODE:
print("\n" + "="*60)
print("TESTING GPU CONTEXT FUNCTIONS")
print("="*60)


def timeit(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        
        print(f"\n{'='*60}")
        print(f"⏱️  {func.__name__} execution time: {elapsed_time*1000:.2f} ms ({elapsed_time:.4f} s)")
        print(f"{'='*60}\n")
        
        return result
    return wrapper

try:
    # Test if functions exist
    print("✓ initialize_gpu_context found:", hasattr(lib, 'initialize_gpu_context'))
    print("✓ update_spline_fast found:", hasattr(lib, 'update_spline_fast'))
    print("✓ update_transforms found:", hasattr(lib, 'update_transforms'))
    print("✓ destroy_gpu_context found:", hasattr(lib, 'destroy_gpu_context'))
except Exception as e:
    print(f"✗ Error checking functions: {e}")

print("="*60 + "\n")

class SimulationWorker(QThread):
    """Background thread for running simulation"""
    finished = Signal(np.ndarray, list, str, dict)  # points, segments, info_text, solution_data
    error = Signal(str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
    
    def run(self):
        try:
            mode = self.params.get('mode', 'FK')
            
            if mode == 'FK':
                points, segments, info = compute_simulation_fk(self.params)
                solution_data = {
                    'pressures': {
                        'A': self.params['channel_a'],
                        'B': self.params['channel_b'],
                        'C': self.params['channel_c'],
                        'D': self.params['channel_d']
                    }
                }
                self.finished.emit(points, segments, info, solution_data)
            
            elif mode == 'IK':
                points, segments, info, solution_data = compute_simulation_ik(self.params)
                self.finished.emit(points, segments, info, solution_data)
                
        except ValueError as e:
            self.error.emit(str(e))
        except Exception as e:
            import traceback
            error_msg = f"!! Unexpected Error:\n{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)

class GPUSplineContext:
    """Manages persistent GPU memory for real-time spline generation"""
    
    def __init__(self):
        self.initialized = False
        self.num_segments = None
        self.resolution = None
        self.total_points = None
        
        # Cached data
        self.length = None
        self.T_cumulative_flat = None
        
        # Store segment rigidities (EI values) for fast curvature computation
        self.segment_EI = None
    
    def initialize(self, segments, resolution, epsilon_pre, etan_value):
        """Initialize GPU context with geometry"""
        print(f"\n[GPU] initialize() called with {len(segments)} segments, resolution={resolution}")
        
        if self.initialized:
            print("[GPU] Already initialized, destroying old context first...")
            self.destroy()
        
        self.num_segments = len(segments)
        self.resolution = resolution
        self.total_points = self.num_segments * resolution
        
        print(f"[GPU] Total points to generate: {self.total_points}")
        
        # Prepare static data
        self.length = np.array([s.length for s in segments], dtype=np.float32)
        print(f"[GPU] Segment lengths: {self.length}")
        
        # Compute segment rigidities (needed for curvature calculation)
        self.segment_EI = np.zeros(self.num_segments, dtype=np.float64)
        for i, seg in enumerate(segments):
            seg.I = fk.compute_second_moment(seg.out_radius, seg.wall_thickness)
            seg.EI = fk.flexural_rigidity(etan_value, seg.I)
            self.segment_EI[i] = seg.EI
        
        print(f"[GPU] Computed EI values: {self.segment_EI}")
        
        # Compute initial transforms (with zero curvature)
        print("[GPU] Computing initial transforms...")
        kappa_init = np.zeros(self.num_segments, dtype=np.float32)
        theta_init = np.zeros(self.num_segments, dtype=np.float32)
        phi_init = np.zeros(self.num_segments, dtype=np.float32)
        
        T_cumulative = spline.compute_cumulative_transforms(
            kappa_init, theta_init, phi_init, self.num_segments
        )
        self.T_cumulative_flat = np.ascontiguousarray(
            T_cumulative.reshape(self.num_segments, 16), 
            dtype=np.float32
        )
        
        print(f"[GPU] T_cumulative shape: {self.T_cumulative_flat.shape}")
        print(f"[GPU] Calling lib.initialize_gpu_context...")
        
        # NEW: Try to initialize GPU context with retry logic
        max_retries = 3
        error_code = -1
        
        for attempt in range(max_retries):
            error_code = lib.initialize_gpu_context(
                self.num_segments,
                self.resolution,
                self.length,
                self.T_cumulative_flat
            )
            
            print(f"[GPU] Attempt {attempt + 1}: lib.initialize_gpu_context returned: {error_code}")
            
            if error_code == 0:
                break  # Success!
            
            # If error 999 (context issue), try to reset GPU
            if error_code == 999:
                print(f"[GPU] Error 999 detected (invalid context), resetting GPU...")
                try:
                    lib.reset_gpu_context()
                    import time
                    time.sleep(0.5)  # Give GPU time to reset
                except Exception as e:
                    print(f"[GPU] Reset failed: {e}")
            
            if attempt < max_retries - 1:
                print(f"[GPU] Retrying in 1 second...")
                import time
                time.sleep(1)
        
        if error_code != 0:
            raise RuntimeError(f"Failed to initialize GPU context: error {error_code}")
        
        self.initialized = True
        print(f"✓ [GPU] Context initialized\n")

    def update_fast(self, kappa, theta, phi):
        """Fast update with new curvatures (GPU resident)"""
        if not self.initialized:
            raise RuntimeError("GPU context not initialized!")
        
        # Prepare arrays
        kappa = np.ascontiguousarray(kappa, dtype=np.float32)
        theta = np.ascontiguousarray(theta, dtype=np.float32)
        phi = np.ascontiguousarray(phi, dtype=np.float32)
        
        # Allocate output
        output = np.zeros(self.total_points * 3, dtype=np.float32)
        
        # Fast GPU update
        error_code = lib.update_spline_fast(kappa, theta, phi, output)
        
        if error_code != 0:
            raise RuntimeError(f"GPU update failed: {error_code}")
        
        return output.reshape(-1, 3)
    
    def update_transforms(self, kappa, theta, phi):
        """Update transformation matrices when curvatures change significantly"""
        if not self.initialized:
            return
        
        T_cumulative = spline.compute_cumulative_transforms(
            kappa, theta, phi, self.num_segments
        )
        self.T_cumulative_flat = np.ascontiguousarray(
            T_cumulative.reshape(self.num_segments, 16), 
            dtype=np.float32
        )
        
        error_code = lib.update_transforms(self.T_cumulative_flat)
        if error_code != 0:
            print(f"Warning: Failed to update transforms: {error_code}")
    
    def destroy(self):
        """Clean up GPU memory"""
        if self.initialized:
            lib.destroy_gpu_context()
            self.initialized = False
            print("✓ GPU context destroyed")
    
    def __del__(self):
        """Destructor - last resort cleanup"""
        if self.initialized:
            print("WARNING: GPU context destroyed in __del__ (should call destroy() explicitly)")
            self.destroy()

class RobotGUI(QMainWindow):

    COLOR_PALETTES = {
        'Neon':         ['#C200FB', '#EC0868', '#FC2F00', '#EC7D10', '#FFBC0A', '#00FFFF', '#3772ff', '#3A0CA3', '#9EF01A', '#16DB65' ],
        'Serenity':     ['#03045E', '#023E8A', '#0077B6', '#0096C7', '#3CA6BB', '#00B4D8', '#52DCFF', '#48CAE4', '#90E0EF', '#ADE8F4'],
        'Nightfall':    ['#10002B', '#240046', '#3C096C', '#5A189A', '#7B2CBF', '#9D4EDD', '#C77DFF', '#E0AAFF', '#F3D2FF', '#FAE8FF'],
        'Deep':         ['#0466C8', '#0353A4', '#023E7D', '#002855', '#001845', '#001233', '#33415C', '#5C677D', '#7D8597', '#979DAC'],
        'Nature':       ['#004B23', '#006400', '#007200', '#008000', '#38B000', '#70E000', '#A0F200', '#D0FF70', '#E0FFA0', '#F0FFC0'],
        'Firestarter':  ['#03071E', '#370617', '#6A040F', '#9D0208', '#D00000', '#DC2F02', '#E85D04', '#F48C06', '#FAA307', '#FFBA08'],
        'Sunset':       ['#001219', '#005F73', '#0A9396', '#94D2BD', '#E9D8A6', '#EE9B00', '#CA6702', '#BB3E03', '#AE2012', '#9B2226']
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Soft Robot Kinematics Simulator")
        self.setGeometry(100, 100, 1800, 1200)
        
        self.material_library = MaterialLibrary()

        # Simulation state
        self.points = None
        self.segments = None
        self.worker = None

        # ============================================================================
        # NEW: GPU context for real-time visualization
        # ============================================================================
        self.gpu_context = GPUSplineContext()
        self.live_preview_active = False
        self.update_timer = None  # For debouncing slider updates
        
        self.live_meshes = {
            'centerline': None,
            'boundaries': None,
            'tubes': []
        }

        # self.plotter.enable_anti_aliasing('fxaa')  # Fast anti-aliasing
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Visualization and output
        right_panel = self.create_display_panel()
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
        
    def create_control_panel(self):
        """Create left control panel with parameter inputs"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("Simulation Parameters")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Tab widget for organized controls
        tabs = QTabWidget()
        tabs.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        # Actuation tab
        actuation_tab = self.create_actuation_controls()
        tabs.addTab(actuation_tab, "Actuation")
        
        # Geometry tab
        geometry_tab = self.create_geometry_controls()
        tabs.addTab(geometry_tab, "Geometry")
        
        # Material tab
        material_tab = self.create_material_controls()
        tabs.addTab(material_tab, "Material")
        
        # Visualisation tab
        visualisation_tab = self.create_visualisation_controls()
        tabs.addTab(visualisation_tab, "Visualisation")

        layout.addWidget(tabs, stretch=1)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.run_button = QPushButton("Run Simulation")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #7132CA;
                color: white;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #471396;
            }
            QPushButton:disabled {
                background-color: #C5BAFF;
            }
        """)
        self.run_button.clicked.connect(self.run_simulation)
        
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.setStyleSheet("""
            QPushButton {
                background-color: #7A7A7A;  /* neutral grey */
                color: white;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5E5E5E;  /* slightly darker on hover */
            }
            QPushButton:disabled {
                background-color: #C0C0C0;  /* light grey when disabled */
            }
        """)
        self.reset_view_button.clicked.connect(self.reset_camera)
        
        self.save_button = QPushButton("Save Screenshot")
        self.save_button.setEnabled(False)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #7A7A7A;  /* neutral grey */
                color: white;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5E5E5E;  /* slightly darker on hover */
            }
            QPushButton:disabled {
                background-color: #C0C0C0;  /* light grey when disabled */
            }
        """)
        self.save_button.clicked.connect(self.save_screenshot)
        
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.reset_view_button)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)
        
        # layout.addStretch()
        
        return panel
    
    def create_actuation_controls(self):
        """Create actuation controls with FK/IK mode switching"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Mode selection at the top
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Actuation Mode:"))
        
        from PySide6.QtWidgets import QComboBox
        self.actuation_mode_combo = QComboBox()
        self.actuation_mode_combo.addItems(["Forward Kinematics", "Inverse Kinematics"])
        self.actuation_mode_combo.currentTextChanged.connect(self.switch_actuation_mode)
        mode_layout.addWidget(self.actuation_mode_combo)
        mode_layout.addStretch()
        
        layout.addLayout(mode_layout)
        
        # Separator
        from PySide6.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Stacked widget to switch between FK and IK controls
        from PySide6.QtWidgets import QStackedWidget
        self.actuation_stack = QStackedWidget()
        
        # FK controls (pressure inputs)
        fk_widget = self.create_fk_controls()
        self.actuation_stack.addWidget(fk_widget)
        
        # IK controls (position inputs)
        ik_widget = self.create_ik_controls()
        self.actuation_stack.addWidget(ik_widget)
        
        layout.addWidget(self.actuation_stack)
        
        return widget
    
    def create_fk_controls(self):
        """Create forward kinematics pressure controls"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        self.pressure_sliders = {}
        channels = ['A', 'B', 'C', 'D']

        for i, ch in enumerate(channels):
            # Label
            label = QLabel(f"Channel {ch} (kPa):")
            layout.addWidget(label, i, 0)

            # Slider
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(1000)
            slider.setValue(60 if ch == 'A' else 0)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(100)
            layout.addWidget(slider, i, 1)

            # LineEdit for precise input
            line_edit = QLineEdit()
            line_edit.setFixedWidth(60)
            line_edit.setText(f"{slider.value()/100:.2f}")
            layout.addWidget(line_edit, i, 2)

            # Bidirectional updates
            def slider_changed(v, le=line_edit):
                le.setText(f"{v/100:.2f}")

            def lineedit_changed(text, s=slider):
                try:
                    val = float(text)
                    s.setValue(int(val*100))
                except ValueError:
                    pass

            slider.valueChanged.connect(slider_changed)
            line_edit.editingFinished.connect(lambda le=line_edit: lineedit_changed(le.text()))
            
            # ============================================================================
            # NEW: Connect to real-time update
            # ============================================================================
            slider.valueChanged.connect(self.on_pressure_changed_realtime)

            self.pressure_sliders[ch] = slider
        
        # ============================================================================
        # NEW: Add live preview checkbox
        # ============================================================================
        live_preview_layout = QHBoxLayout()
        self.live_preview_checkbox = QCheckBox("🔴 Live Preview (GPU)")
        self.live_preview_checkbox.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                color: #7132CA;
            }
            QCheckBox::indicator:checked {
                background-color: #00FF00;
            }
        """)
        self.live_preview_checkbox.stateChanged.connect(self.toggle_live_preview)
        live_preview_layout.addWidget(self.live_preview_checkbox)
        
        self.live_fps_label = QLabel("FPS: --")
        self.live_fps_label.setStyleSheet("color: gray; font-size: 9pt;")
        live_preview_layout.addWidget(self.live_fps_label)
        live_preview_layout.addStretch()
        
        layout.addLayout(live_preview_layout, 4, 0, 1, 3)
        
        layout.setRowStretch(6, 1)
        return widget
        
    def create_ik_controls(self):
        """Create inverse kinematics position controls"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Target position inputs
        layout.addWidget(QLabel("Target End-Effector Position:"), 0, 0, 1, 2)
        
        layout.addWidget(QLabel("X (mm):"), 1, 0)
        self.target_x = QDoubleSpinBox()
        self.target_x.setRange(-500, 500)
        self.target_x.setValue(0)
        self.target_x.setSingleStep(5.0)
        layout.addWidget(self.target_x, 1, 1)
        
        layout.addWidget(QLabel("Y (mm):"), 2, 0)
        self.target_y = QDoubleSpinBox()
        self.target_y.setRange(-500, 500)
        self.target_y.setValue(0)
        self.target_y.setSingleStep(5.0)
        layout.addWidget(self.target_y, 2, 1)
        
        layout.addWidget(QLabel("Z (mm):"), 3, 0)
        self.target_z = QDoubleSpinBox()
        self.target_z.setRange(0, 500)
        self.target_z.setValue(100)
        self.target_z.setSingleStep(5.0)
        layout.addWidget(self.target_z, 3, 1)
        
        # Optional: Orientation constraints (can be added later)
        from PySide6.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line, 4, 0, 1, 2)
        
        # IK solver parameters
        layout.addWidget(QLabel("IK Solver Settings:"), 5, 0, 1, 2)
        
        layout.addWidget(QLabel("Max Iterations:"), 6, 0)
        self.ik_max_iter = QSpinBox()
        self.ik_max_iter.setRange(10, 500)
        self.ik_max_iter.setValue(100)
        self.ik_max_iter.setSingleStep(10)
        layout.addWidget(self.ik_max_iter, 6, 1)
        
        layout.addWidget(QLabel("Tolerance (mm):"), 7, 0)
        self.ik_tolerance = QDoubleSpinBox()
        self.ik_tolerance.setRange(0.001, 10.0)
        self.ik_tolerance.setValue(1.0)
        self.ik_tolerance.setSingleStep(0.1)
        self.ik_tolerance.setDecimals(3)
        layout.addWidget(self.ik_tolerance, 7, 1)
        
        layout.addWidget(QLabel("Damping λ:"), 8, 0)
        self.ik_damping = QDoubleSpinBox()
        self.ik_damping.setRange(0.001, 1.0)
        self.ik_damping.setValue(0.01)
        self.ik_damping.setSingleStep(0.01)
        self.ik_damping.setDecimals(3)
        layout.addWidget(self.ik_damping, 8, 1)
        
        layout.setRowStretch(9, 1)
        return widget
    
    def switch_actuation_mode(self, mode):
        """Switch between FK and IK modes"""
        if mode == "Forward Kinematics":
            self.actuation_stack.setCurrentIndex(0)
            self.run_button.setText("Run Forward Kinematics")
        elif mode == "Inverse Kinematics":
            self.actuation_stack.setCurrentIndex(1)
            self.run_button.setText("Run Inverse Kinematics")





    def create_geometry_controls(self):
        """Create segment geometry controls"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Channel geometry
        channel_group = QGroupBox("Channel Geometry")
        channel_layout = QGridLayout(channel_group)
        
        channel_layout.addWidget(QLabel("Channel Radius (mm):"), 0, 0)
        self.channel_radius = QDoubleSpinBox()
        self.channel_radius.setRange(1.0, 10.0)
        self.channel_radius.setValue(5.0)
        self.channel_radius.setSingleStep(0.5)
        channel_layout.addWidget(self.channel_radius, 0, 1)
        
        channel_layout.addWidget(QLabel("Septum Thickness (mm):"), 1, 0)
        self.septum_thickness = QDoubleSpinBox()
        self.septum_thickness.setRange(0.1, 2.0)
        self.septum_thickness.setValue(0.8)
        self.septum_thickness.setSingleStep(0.1)
        channel_layout.addWidget(self.septum_thickness, 1, 1)
        


        self.channel_radius.valueChanged.connect(
            lambda _: self.update_segment_controls(
                self.channel_radius.value(),
                self.septum_thickness.value()
            )
        )

        self.septum_thickness.valueChanged.connect(
            lambda _: self.update_segment_controls(
                self.channel_radius.value(),
                self.septum_thickness.value()
            )
        )
        layout.addWidget(channel_group)
        
        # Number of segments control
        num_seg_layout = QHBoxLayout()
        num_seg_layout.addWidget(QLabel("Number of Segments:"))
        self.num_segments_spin = QSpinBox()
        self.num_segments_spin.setRange(1, 10)
        self.num_segments_spin.setValue(5)
        self.num_segments_spin.valueChanged.connect(
            lambda _: self.update_segment_controls(
                self.channel_radius.value(),
                self.septum_thickness.value()
            )
        )
        self.num_segments_spin.valueChanged.connect(
            lambda: self.update_color_preview())
        num_seg_layout.addWidget(self.num_segments_spin)
        num_seg_layout.addStretch()
        layout.addLayout(num_seg_layout)
        
        # Scrollable segment parameters
        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(500)
        
        self.seg_controls_widget = QWidget()
        self.seg_controls_layout = QVBoxLayout(self.seg_controls_widget)
        scroll.setWidget(self.seg_controls_widget)
        
        layout.addWidget(scroll)
        
        # Initialize segment controls
        self.segment_params = []  # Will store dicts with {'length', 'radius', 'thickness'}
        self.update_segment_controls(self.channel_radius.value(), self.septum_thickness.value() )
        
        return widget
    

    def update_segment_controls(self,channel_radius, septum_thickness):
        while self.seg_controls_layout.count():
            item = self.seg_controls_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.segment_params.clear()
        # channel_radius = self.channel_radius.value()
        # septum_thickness = self.septum_thickness.value()        
        num_segments = self.num_segments_spin.value()
        default_lengths = [40, 50, 45, 55, 30, 35, 40, 45, 50, 55]
        default_radii = [8, 10, 10, 15, 5, 8, 10, 12, 8, 6]
        default_walls = [1, 1, 1, 1.5, 1, 1, 1.2, 1, 1, 1]
        min_radius = 0.707 * (2*channel_radius + septum_thickness) + 1.0  

        for i in range(num_segments):
            seg_group = QGroupBox(f"Segment {i+1}")
            seg_layout = QGridLayout(seg_group)
            
            # Length
            seg_layout.addWidget(QLabel("Length (mm):"), 0, 0)
            length_spin = QDoubleSpinBox()
            length_spin.setRange(10.0, 150.0)
            length_spin.setValue(default_lengths[i])
            length_spin.setSingleStep(5.0)
            seg_layout.addWidget(length_spin, 0, 1)
            
            # Outer Radius
            seg_layout.addWidget(QLabel("Outer Radius (mm):"), 1, 0)
            radius_spin = QDoubleSpinBox()
            radius_spin.setRange(min_radius, 20.0)
            radius_spin.setValue(default_radii[i])
            radius_spin.setSingleStep(0.5)
            seg_layout.addWidget(radius_spin, 1, 1)
            
            # Wall Thickness
            seg_layout.addWidget(QLabel("Wall Thick. (mm):"), 2, 0)
            thickness_spin = QDoubleSpinBox()
            thickness_spin.setRange(0.05*min_radius, 0.15*min_radius)
            thickness_spin.setValue(default_walls[i])
            thickness_spin.setSingleStep(0.1)
            seg_layout.addWidget(thickness_spin, 2, 1)
            
            self.seg_controls_layout.addWidget(seg_group)
            
            self.segment_params.append({
                'length': length_spin,
                'radius': radius_spin,
                'thickness': thickness_spin
            })

    def create_material_controls(self):
        """Create material property controls with preset library"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        # Material preset selection
        preset_group = QGroupBox("Material Presets")
        preset_layout = QVBoxLayout(preset_group)
        
        preset_select_layout = QHBoxLayout()
        preset_select_layout.addWidget(QLabel("Select Preset:"))
        
        from PySide6.QtWidgets import QComboBox
        self.material_preset_combo = QComboBox()
        self.material_preset_combo.addItem("-- Custom --")
        self.material_preset_combo.addItems(self.material_library.list_materials())
        self.material_preset_combo.currentTextChanged.connect(self.load_material_preset)
        preset_select_layout.addWidget(self.material_preset_combo)
        preset_layout.addLayout(preset_select_layout)
        
        # Preset management buttons
        preset_buttons_layout = QHBoxLayout()
        
        self.save_preset_button = QPushButton("Save as Preset")
        self.save_preset_button.clicked.connect(self.save_material_preset)
        preset_buttons_layout.addWidget(self.save_preset_button)
        
        self.delete_preset_button = QPushButton("Delete Preset")
        self.delete_preset_button.clicked.connect(self.delete_material_preset)
        preset_buttons_layout.addWidget(self.delete_preset_button)
        
        preset_layout.addLayout(preset_buttons_layout)
        layout.addWidget(preset_group)
        
        # Separator
        from PySide6.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Material parameters grid
        params_grid = QGridLayout()
        
        params_grid.addWidget(QLabel("Pre-strain (ε):"), 0, 0)
        self.prestrain = QDoubleSpinBox()
        self.prestrain.setRange(0.0, 0.5)
        self.prestrain.setValue(0.15)
        self.prestrain.setSingleStep(0.01)
        self.prestrain.setDecimals(3)
        self.prestrain.valueChanged.connect(self.mark_custom_material)
        params_grid.addWidget(self.prestrain, 0, 1)
        
        params_grid.addWidget(QLabel("Bulk Modulus (MPa):"), 1, 0)
        self.bulk_modulus = QDoubleSpinBox()
        self.bulk_modulus.setRange(10, 200)
        self.bulk_modulus.setValue(50)
        self.bulk_modulus.setSingleStep(10)
        self.bulk_modulus.valueChanged.connect(self.mark_custom_material)
        params_grid.addWidget(self.bulk_modulus, 1, 1)
        
        params_grid.addWidget(QLabel("Material Model:"), 2, 0)
        self.material_model = QComboBox()
        self.material_model.addItems(["Neo-Hookean", "Mooney-Rivlin", "Ogden"])
        self.material_model.currentTextChanged.connect(self.update_material_parameters)
        # Note: Model selection does NOT mark as custom
        params_grid.addWidget(self.material_model, 2, 1)
        
        layout.addLayout(params_grid)
        
        # Model-specific parameters (same as before)
        model_params_layout = QGridLayout()
        
        self.mu_label = QLabel("μ (MPa):")
        self.mu_spin = QDoubleSpinBox()
        self.mu_spin.setRange(0.01, 5.0)
        self.mu_spin.setSingleStep(0.05)
        self.mu_spin.setValue(0.5)
        # Note: Model parameters do NOT mark as custom
        
        model_params_layout.addWidget(self.mu_label, 0, 0)
        model_params_layout.addWidget(self.mu_spin, 0, 1)
        
        self.c1_label = QLabel("C1 (MPa):")
        self.c1_spin = QDoubleSpinBox()
        self.c1_spin.setRange(0.001, 5.0)
        self.c1_spin.setSingleStep(0.01)
        self.c1_spin.setValue(0.3)
        
        self.c2_label = QLabel("C2 (MPa):")
        self.c2_spin = QDoubleSpinBox()
        self.c2_spin.setRange(0.001, 5.0)
        self.c2_spin.setSingleStep(0.01)
        self.c2_spin.setValue(0.1)
        
        model_params_layout.addWidget(self.c1_label, 1, 0)
        model_params_layout.addWidget(self.c1_spin, 1, 1)
        model_params_layout.addWidget(self.c2_label, 2, 0)
        model_params_layout.addWidget(self.c2_spin, 2, 1)
        
        self.og_mu_label = QLabel("μ (MPa):")
        self.og_mu_spin = QDoubleSpinBox()
        self.og_mu_spin.setRange(0.01, 5.0)
        self.og_mu_spin.setSingleStep(0.05)
        self.og_mu_spin.setValue(0.5)
        
        self.alpha_label = QLabel("α:")
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(1.0, 15.0)
        self.alpha_spin.setSingleStep(0.5)
        self.alpha_spin.setValue(8.0)
        
        model_params_layout.addWidget(self.og_mu_label, 3, 0)
        model_params_layout.addWidget(self.og_mu_spin, 3, 1)
        model_params_layout.addWidget(self.alpha_label, 4, 0)
        model_params_layout.addWidget(self.alpha_spin, 4, 1)
        
        layout.addLayout(model_params_layout)
        
        # Initialize UI state
        self.update_material_parameters(self.material_model.currentText())
        
        layout.addStretch()
        return widget

    # Add these new methods to RobotGUI class:

    def mark_custom_material(self):
        """Mark material as custom when parameters change"""
        # Only mark as custom when epsilon_pre or bulk_modulus changes
        # Mathematical model choice doesn't affect material preset
        sender = self.sender()
        if sender in [self.prestrain, self.bulk_modulus]:
            if self.material_preset_combo.currentText() != "-- Custom --":
                self.material_preset_combo.blockSignals(True)
                self.material_preset_combo.setCurrentText("-- Custom --")
                self.material_preset_combo.blockSignals(False)

    def load_material_preset(self, preset_name):
        """Load material parameters from preset"""
        if preset_name == "-- Custom --":
            return
        
        material_data = self.material_library.get_material(preset_name)
        if not material_data:
            return
        
        # Block signals to prevent marking as custom during load
        self.prestrain.blockSignals(True)
        self.bulk_modulus.blockSignals(True)
        
        # Load only basic material properties (not mathematical model)
        self.prestrain.setValue(material_data['epsilon_pre'])
        self.bulk_modulus.setValue(material_data['bulk_modulus'] / 1e6)  # Convert to MPa
        
        # Mathematical model and its parameters are chosen by user
        # and remain unchanged when loading a material preset
        
        self.prestrain.blockSignals(False)
        self.bulk_modulus.blockSignals(False)
        
        self.output_text.append(f"✓ Loaded material preset: {preset_name}")

    def save_material_preset(self):
        """Save current material parameters as a preset"""
        from PySide6.QtWidgets import QInputDialog
        
        name, ok = QInputDialog.getText(
            self, 
            "Save Material Preset",
            "Enter preset name:"
        )
        
        if not ok or not name.strip():
            return
        
        name = name.strip()
        
        # Save only material properties (not mathematical model)
        self.material_library.add_material(
            name=name,
            epsilon_pre=self.prestrain.value(),
            bulk_modulus=self.bulk_modulus.value() * 1e6
        )
        
        # Update combo box
        self.material_preset_combo.blockSignals(True)
        self.material_preset_combo.clear()
        self.material_preset_combo.addItem("-- Custom --")
        self.material_preset_combo.addItems(self.material_library.list_materials())
        self.material_preset_combo.setCurrentText(name)
        self.material_preset_combo.blockSignals(False)
        
        self.output_text.append(f"✓ Saved material preset: {name}")
        self.output_text.append(f"  (ε_pre={self.prestrain.value():.3f}, K={self.bulk_modulus.value():.0f} MPa)")

    def delete_material_preset(self):
        """Delete selected material preset"""
        preset_name = self.material_preset_combo.currentText()
        
        if preset_name == "-- Custom --":
            self.output_text.append("✗ Cannot delete custom settings")
            return
        
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "Delete Preset",
            f"Are you sure you want to delete '{preset_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.material_library.remove_material(preset_name):
                # Update combo box
                self.material_preset_combo.blockSignals(True)
                self.material_preset_combo.clear()
                self.material_preset_combo.addItem("-- Custom --")
                self.material_preset_combo.addItems(self.material_library.list_materials())
                self.material_preset_combo.setCurrentText("-- Custom --")
                self.material_preset_combo.blockSignals(False)
                
                self.output_text.append(f"✓ Deleted material preset: {preset_name}")
            else:
                self.output_text.append(f"✗ Failed to delete preset: {preset_name}")

    def block_material_signals(self, block):
        """Block/unblock signals for material property widgets (not model params)"""
        widgets = [self.prestrain, self.bulk_modulus]
        for widget in widgets:
            widget.blockSignals(block)


    def update_material_parameters(self, name):
        """Show relevant fields + set default values"""
        
        for w in [
            self.mu_label, self.mu_spin,
            self.c1_label, self.c1_spin,
            self.c2_label, self.c2_spin,
            self.og_mu_label, self.og_mu_spin,
            self.alpha_label, self.alpha_spin
        ]:
            w.hide()

        if name == "Neo-Hookean":
            self.mu_label.show()
            self.mu_spin.show()
            self.mu_spin.setValue(0.5)

        elif name == "Mooney-Rivlin":
            self.c1_label.show()
            self.c1_spin.show()
            self.c2_label.show()
            self.c2_spin.show()
            self.c1_spin.setValue(0.3)
            self.c2_spin.setValue(0.1)

        elif name == "Ogden":
            self.og_mu_label.show()
            self.og_mu_spin.show()
            self.alpha_label.show()
            self.alpha_spin.show()
            self.og_mu_spin.setValue(0.5)
            self.alpha_spin.setValue(8.0)

    def create_visualisation_controls(self):
        """Create visualisation property controls"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Main controls grid
        controls_grid = QGridLayout()
        
        # Resolution control
        controls_grid.addWidget(QLabel("Spline Resolution:"), 0, 0)
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(50, 2000)
        self.resolution_spin.setValue(50)
        self.resolution_spin.setSingleStep(100)
        self.resolution_spin.setToolTip("Number of points per segment for visualization")
        controls_grid.addWidget(self.resolution_spin, 0, 1)
        
        # Color palette selection
        controls_grid.addWidget(QLabel("Color Palette:"), 1, 0)
        from PySide6.QtWidgets import QComboBox
        self.color_palette = QComboBox()
        self.color_palette.addItems([
            "Neon",
            "Serenity",
            "Nightfall", 
            "Deep",
            "Nature",
            "Firestarter",
            "Sunset"
        ])
        self.color_palette.setToolTip("Color scheme for segment tubes")
        self.color_palette.currentTextChanged.connect(self.update_color_preview)
        controls_grid.addWidget(self.color_palette, 1, 1)
        
        # Grid visibility
        controls_grid.addWidget(QLabel("Show Grid:"), 2, 0)
        from PySide6.QtWidgets import QCheckBox
        self.show_grid = QCheckBox()
        self.show_grid.setChecked(True)
        controls_grid.addWidget(self.show_grid, 2, 1)
        
        # Show segment boundaries
        controls_grid.addWidget(QLabel("Show Segment Ends:"), 3, 0)
        self.show_boundaries = QCheckBox()
        self.show_boundaries.setChecked(True)
        controls_grid.addWidget(self.show_boundaries, 3, 1)
        
        # Tube opacity
        controls_grid.addWidget(QLabel("Tube Opacity:"), 4, 0)
        self.tube_opacity = QDoubleSpinBox()
        self.tube_opacity.setRange(0.1, 1.0)
        self.tube_opacity.setValue(0.6)
        self.tube_opacity.setSingleStep(0.1)
        controls_grid.addWidget(self.tube_opacity, 4, 1)
        
        layout.addLayout(controls_grid)
        
        preview_group = QGroupBox("Color Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_grid = QGridLayout()
        
        # Centerline color
        preview_grid.addWidget(QLabel("Centerline:"), 0, 0)
        self.centerline_color_label = QLabel()
        self.centerline_color_label.setFixedSize(80, 25)
        self.centerline_color_label.setStyleSheet("background-color: blue; border: 1px solid black;")
        preview_grid.addWidget(self.centerline_color_label, 0, 1)
        
        preview_grid.addWidget(QLabel("Segment Ends:"), 1, 0)
        self.ends_color_label = QLabel()
        self.ends_color_label.setFixedSize(80, 25)
        self.ends_color_label.setStyleSheet("background-color: red; border: 1px solid black;")
        preview_grid.addWidget(self.ends_color_label, 1, 1)
        
        preview_layout.addLayout(preview_grid)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(300)
        self.segment_colors_widget = QWidget()
        self.segment_colors_layout = QVBoxLayout(self.segment_colors_widget)
        scroll.setWidget(self.segment_colors_widget)
        
        preview_layout.addWidget(scroll) 
        
        layout.addWidget(preview_group)
        
        # Initialize color preview
        self.update_color_preview()
        
        layout.addStretch()
        return widget
    
    def update_color_preview(self):
        """Update the color preview based on current palette and number of segments"""
        # Clear existing preview - improved cleanup
        while self.segment_colors_layout.count():
            item = self.segment_colors_layout.takeAt(0)
            if item.layout():
                # If it's a layout, clear its widgets first
                while item.layout().count():
                    widget_item = item.layout().takeAt(0)
                    if widget_item.widget():
                        widget_item.widget().deleteLater()
                item.layout().deleteLater()
            elif item.widget():
                item.widget().deleteLater()
        
        # Get current palette and number of segments
        palette_name = self.color_palette.currentText()
        colors = self.COLOR_PALETTES[palette_name]
        num_segments = self.num_segments_spin.value()
        
        # Create color swatches for each segment
        for i in range(num_segments):
            seg_layout = QHBoxLayout()
            
            label = QLabel(f"Segment {i+1}:")
            label.setFixedWidth(80)
            seg_layout.addWidget(label)
            
            color_swatch = QLabel()
            color_swatch.setFixedSize(80, 25)
            color = colors[i % len(colors)]
            color_swatch.setStyleSheet(f"background-color: {color}; border: 1px solid black;")
            seg_layout.addWidget(color_swatch)
            
            # Add hex code
            hex_label = QLabel(color)
            hex_label.setStyleSheet("color: gray; font-size: 9pt;")
            seg_layout.addWidget(hex_label)
            
            seg_layout.addStretch()
            self.segment_colors_layout.addLayout(seg_layout)
        
        # Add a stretch at the end to push everything to the top
        self.segment_colors_layout.addStretch()

    def create_display_panel(self):
        """Create right display panel with interactive PyVista viewer"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create vertical splitter for resizable visualization and output
        splitter = QSplitter(Qt.Vertical)
        
        # Interactive PyVista viewer
        self.plotter = QtInteractor(panel)
        self.plotter.set_background('white')
        
        # Add initial text
        text_actor = self.plotter.add_text(
            "Please run the simulation first",
            font_size=12,
            color='black'
        )
        
        splitter.addWidget(self.plotter.interactor)
        
        # Output text area
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Courier", 9))
        output_layout.addWidget(self.output_text)
        
        splitter.addWidget(output_group)
        
        # Set initial sizes (visualization gets more space)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        
        return panel
    
    def run_simulation(self):
        """Main simulation entry point - routes to FK or IK"""
        mode = self.actuation_mode_combo.currentText()
        
        if mode == "Forward Kinematics":
            self.run_forward_kinematics()
        elif mode == "Inverse Kinematics":
            self.run_inverse_kinematics()


    def run_forward_kinematics(self):
        """Run forward kinematics simulation"""
        self.run_button.setEnabled(False)
        self.output_text.clear()
        self.output_text.append("Running Forward Kinematics...\n")
        
        params = self.build_simulation_params()
        params['mode'] = 'FK'
        
        # Run in background thread
        self.worker = SimulationWorker(params)
        self.worker.finished.connect(self.on_simulation_complete)
        self.worker.error.connect(self.on_simulation_error)
        self.worker.start()


    def run_inverse_kinematics(self):
        """Run inverse kinematics simulation"""
        self.run_button.setEnabled(False)
        self.output_text.clear()
        self.output_text.append("Running Inverse Kinematics...\n")
        
        params = self.build_simulation_params()
        params['mode'] = 'IK'
        
        # Add IK-specific parameters
        params['target_x'] = self.target_x.value() * 1e-3  # mm to m
        params['target_y'] = self.target_y.value() * 1e-3
        params['target_z'] = self.target_z.value() * 1e-3
        params['ik_max_iter'] = self.ik_max_iter.value()
        params['ik_tolerance'] = self.ik_tolerance.value() * 1e-3  # mm to m
        params['ik_damping'] = self.ik_damping.value()
        
        # Run in background thread
        self.worker = SimulationWorker(params)
        self.worker.finished.connect(self.on_simulation_complete)
        self.worker.error.connect(self.on_simulation_error)
        self.worker.start()


    def build_simulation_params(self):
        """Build common simulation parameters"""
        params = {
            'channel_radius': self.channel_radius.value() * 1e-3,
            'septum_thickness': self.septum_thickness.value() * 1e-3,
            'num_segments': self.num_segments_spin.value(),
            'segment_lengths': [s['length'].value() * 1e-3 for s in self.segment_params],
            'segment_radii': [s['radius'].value() * 1e-3 for s in self.segment_params],
            'segment_thickness': [s['thickness'].value() * 1e-3 for s in self.segment_params],
            'epsilon_pre': self.prestrain.value(),
            'bulk_modulus': self.bulk_modulus.value() * 1e6,
            'material_model': self.material_model.currentText(),
            'resolution': self.resolution_spin.value(),
        }
        
        # Add material model-specific parameters
        if self.material_model.currentText() == "Neo-Hookean":
            params['mu'] = self.mu_spin.value() * 1e6
        elif self.material_model.currentText() == "Mooney-Rivlin":
            params['c1'] = self.c1_spin.value() * 1e6
            params['c2'] = self.c2_spin.value() * 1e6
        elif self.material_model.currentText() == "Ogden":
            params['mu'] = self.og_mu_spin.value() * 1e6
            params['alpha'] = self.alpha_spin.value()
        
        # Add pressure inputs (used for FK, or as initial guess for IK)
        params['channel_a'] = self.pressure_sliders['A'].value() * 1e3  # Convert to Pa
        params['channel_b'] = self.pressure_sliders['B'].value() * 1e3
        params['channel_c'] = self.pressure_sliders['C'].value() * 1e3
        params['channel_d'] = self.pressure_sliders['D'].value() * 1e3
        
        return params



    # def run_simulation(self):
    #     self.run_button.setEnabled(False)
    #     self.output_text.clear()
    #     self.output_text.append("Running simulation...\n")
        
    #     if self.material_model.currentText() == "Neo-Hookean":
    #         params = {
    #             'channel_a': self.pressure_sliders['A'].value() * 1e3,
    #             'channel_b': self.pressure_sliders['B'].value() * 1e3,
    #             'channel_c': self.pressure_sliders['C'].value() * 1e3,
    #             'channel_d': self.pressure_sliders['D'].value() * 1e3,
    #             'channel_radius': self.channel_radius.value() * 1e-3,
    #             'septum_thickness': self.septum_thickness.value() * 1e-3,
    #             'num_segments': self.num_segments_spin.value(),
    #             'segment_lengths': [s['length'].value() * 1e-3 for s in self.segment_params],
    #             'segment_radii': [s['radius'].value() * 1e-3 for s in self.segment_params],
    #             'segment_thickness': [s['thickness'].value() * 1e-3 for s in self.segment_params],
    #             'epsilon_pre': self.prestrain.value(),
    #             'bulk_modulus': self.bulk_modulus.value() * 1e6,
    #             'material_model': "Neo-Hookean",
    #             'mu': self.mu_spin.value() * 1e6,
    #             'resolution': self.resolution_spin.value(),  # Add resolution
    #         }

    #     if self.material_model.currentText() == "Mooney-Rivlin":
    #         params = {
    #             'channel_a': self.pressure_sliders['A'].value() * 1e3,
    #             'channel_b': self.pressure_sliders['B'].value() * 1e3,
    #             'channel_c': self.pressure_sliders['C'].value() * 1e3,
    #             'channel_d': self.pressure_sliders['D'].value() * 1e3,
    #             'channel_radius': self.channel_radius.value() * 1e-3,
    #             'septum_thickness': self.septum_thickness.value() * 1e-3,
    #             'num_segments': self.num_segments_spin.value(),
    #             'segment_lengths': [s['length'].value() * 1e-3 for s in self.segment_params],
    #             'segment_radii': [s['radius'].value() * 1e-3 for s in self.segment_params],
    #             'segment_thickness': [s['thickness'].value() * 1e-3 for s in self.segment_params],
    #             'epsilon_pre': self.prestrain.value(),
    #             'bulk_modulus': self.bulk_modulus.value() * 1e6,
    #             'material_model': "Mooney-Rivlin",
    #             'c1': self.c1_spin.value() * 1e6,
    #             'c2': self.c2_spin.value() * 1e6,
    #             'resolution': self.resolution_spin.value(),  # Add resolution
    #         }
    #     if self.material_model.currentText() == "Ogden":
    #         params = {
    #             'channel_a': self.pressure_sliders['A'].value() * 1e3,
    #             'channel_b': self.pressure_sliders['B'].value() * 1e3,
    #             'channel_c': self.pressure_sliders['C'].value() * 1e3,
    #             'channel_d': self.pressure_sliders['D'].value() * 1e3,
    #             'channel_radius': self.channel_radius.value() * 1e-3,
    #             'septum_thickness': self.septum_thickness.value() * 1e-3,
    #             'num_segments': self.num_segments_spin.value(),
    #             'segment_lengths': [s['length'].value() * 1e-3 for s in self.segment_params],
    #             'segment_radii': [s['radius'].value() * 1e-3 for s in self.segment_params],
    #             'segment_thickness': [s['thickness'].value() * 1e-3 for s in self.segment_params],
    #             'epsilon_pre': self.prestrain.value(),
    #             'bulk_modulus': self.bulk_modulus.value() * 1e6,
    #             'material_model': "Ogden",
    #             'mu': self.og_mu_spin.value() * 1e6,
    #             'alpha': self.alpha_spin.value(),
    #             'resolution': self.resolution_spin.value(),  # Add resolution
    #         }

    #     # Run in background thread
    #     self.worker = SimulationWorker(params)
    #     self.worker.finished.connect(self.on_simulation_complete)
    #     self.worker.error.connect(self.on_simulation_error)
    #     self.worker.start()
        
    def on_simulation_complete(self, points, segments, info_text, solution_data):
        """Handle completed simulation"""
        self.points = points
        self.segments = segments
        # Display output text
        self.output_text.append(info_text)
        
        # Display solution-specific information
        mode = self.actuation_mode_combo.currentText()
        if mode == "Inverse Kinematics" and 'pressures' in solution_data:
            self.output_text.append("\n=== IK Solution Pressures ===")
            for channel, pressure in solution_data['pressures'].items():
                self.output_text.append(f"Channel {channel}: {pressure/1e3:.2f} kPa")
            
            if solution_data.get('converged', False):
                self.output_text.append(f"\n✓ Converged in {solution_data.get('iterations', 0)} iterations")
            else:
                self.output_text.append(f"\n⚠ Did not converge (max iterations reached)")
            
            # Update pressure sliders to show solution
            self.update_pressure_sliders_from_solution(solution_data['pressures'])
        
        self.output_text.append(f"\n✓ Simulation complete! Generated {len(points)} points")
        
        # Update visualization
        self.update_visualization()
        
        self.run_button.setEnabled(True)
        self.save_button.setEnabled(True)


    def update_pressure_sliders_from_solution(self, pressures):
        """Update FK pressure sliders with IK solution"""
        self.pressure_sliders['A'].blockSignals(True)
        self.pressure_sliders['B'].blockSignals(True)
        self.pressure_sliders['C'].blockSignals(True)
        self.pressure_sliders['D'].blockSignals(True)
        
        self.pressure_sliders['A'].setValue(int(pressures['A'] / 10))
        self.pressure_sliders['B'].setValue(int(pressures['B'] / 10))
        self.pressure_sliders['C'].setValue(int(pressures['C'] / 10))
        self.pressure_sliders['D'].setValue(int(pressures['D'] / 10))
        
        self.pressure_sliders['A'].blockSignals(False)
        self.pressure_sliders['B'].blockSignals(False)
        self.pressure_sliders['C'].blockSignals(False)
        self.pressure_sliders['D'].blockSignals(False)
    
    def on_simulation_error(self, error_msg):
        """Handle simulation error"""
        self.output_text.append(f"\n✗ Error: {error_msg}")
        print(error_msg)
        self.run_button.setEnabled(True)
    
    def update_visualization(self):
        """Update interactive 3D visualization"""
        if self.points is None:
            return
        
        try:
            # Clear previous visualization
            self.plotter.clear()
            self.plotter.set_background('white')
            
            # Add grid (conditionally)
            if self.show_grid.isChecked():
                extent = 1.0
                spacing = 0.05
                coords = np.arange(-extent, extent + spacing, spacing)
                grid_xy = pv.RectilinearGrid(coords, coords, [0.0])
                self.plotter.add_mesh(grid_xy, color="black", style="wireframe", opacity=1.0)
            
            # Generate segment boundaries
            num_segments = len(self.segments)
            resolution = self.resolution_spin.value()
            segment_boundaries = np.arange(0, len(self.points), resolution)
            
            # Draw centerline
            centerline = pv.Spline(self.points, resolution)
            self.plotter.add_mesh(
                centerline,
                color='blue',
                line_width=3,
                # label='Centerline'
            )
            
            # Draw segment boundaries (conditionally)
            if self.show_boundaries.isChecked():
                boundary_points = self.points[segment_boundaries]
                point_cloud = pv.PolyData(boundary_points)
                self.plotter.add_mesh(
                    point_cloud,
                    color='red',
                    point_size=15,
                    render_points_as_spheres=True,
                    # label='Segment Ends'
                )
            
            # Draw segment tubes with selected color palette
            self._draw_segment_tubes_in_plotter(segment_boundaries)
            
            # Add coordinate frame
            scale = 0.02
            position = np.array([0, 0, 0])
            
            arrow_x = pv.Arrow(start=position, direction=[1, 0, 0], scale=scale)
            self.plotter.add_mesh(arrow_x, color='red')
            
            arrow_y = pv.Arrow(start=position, direction=[0, 1, 0], scale=scale)
            self.plotter.add_mesh(arrow_y, color='green')
            
            arrow_z = pv.Arrow(start=position, direction=[0, 0, 1], scale=scale)
            self.plotter.add_mesh(arrow_z, color='blue')
            
            # Add legend
            # self.plotter.add_legend()
            self.plotter.reset_camera()

            self.plotter.enable_parallel_projection()
            self.plotter.camera.parallel_scale = 0.2
            
        except Exception as e:
            self.output_text.append(f"\nVisualization error: {str(e)}")

    def reset_camera(self):
        """Reset camera view to default"""
        if self.points is not None:
            self.plotter.reset_camera()
            self.plotter.enable_parallel_projection()
            self.plotter.camera.parallel_scale = 0.2


    def save_screenshot(self):
        """Save current visualization"""
        if self.points is not None:
            filename = 'robot_3d_view.png'
            
            try:
                # Save screenshot directly from current plotter
                self.plotter.screenshot(filename)
                self.output_text.append(f"\n✓ Screenshot saved as {filename}")
            except Exception as e:
                self.output_text.append(f"\n✗ Screenshot error: {str(e)}")

    def _draw_segment_tubes_in_plotter(self, boundaries):
        """Draw cylindrical tubes representing segment geometry"""
        # Get selected color palette
        palette_name = self.color_palette.currentText()
        colors = self.COLOR_PALETTES[palette_name]
        opacity = self.tube_opacity.value()
        
        for i, seg in enumerate(self.segments):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1] if i < len(self.segments)-1 else len(self.points)
            
            segment_points = self.points[start_idx:end_idx]
            
            # Create tube along segment path
            spline = pv.Spline(segment_points, 50)
            tube = spline.tube(radius=seg.out_radius, n_sides=12)
            
            self.plotter.add_mesh(
                tube,
                color=colors[i % len(colors)],
                opacity=opacity,
                # label=f'Segment {i+1}'
            )
    # ============================================================================
    # REAL-TIME GPU VISUALIZATION METHODS
    # ============================================================================
    
    def toggle_live_preview(self, state):
        """Enable/disable live preview mode"""
        print(f"\n[DEBUG] toggle_live_preview called with state: {state} (Qt.Checked={Qt.Checked})")  # ADD THIS
        
        if state == 2:  # Qt.Checked value
            print("[DEBUG] Enabling live preview...")  # ADD THIS
            # Enable live preview
            self.enable_live_preview()
        else:
            print("[DEBUG] Disabling live preview...")  # ADD THIS
            # Disable live preview
            self.disable_live_preview()
    
    def enable_live_preview(self):
        """Initialize GPU context and enable real-time updates"""
        print("\n[DEBUG] enable_live_preview() called")
        
        try:
            self.output_text.append("\n🔄 Initializing GPU context for live preview...")
            
            # Check GPU health first
            print("[DEBUG] Checking GPU status...")
            gpu_status = lib.check_gpu_status()
            if gpu_status != 0:
                raise RuntimeError(
                    f"GPU is not available or not working properly (error {gpu_status})\n"
                    "Please check:\n"
                    "  • NVIDIA drivers are installed\n"
                    "  • CUDA is properly configured\n"
                    "  • GPU is not being used by another process"
                )
            
            # Get current parameters
            params = self.build_simulation_params()
            
            # Create segments
            segments = []
            for i, length in enumerate(params['segment_lengths']):
                seg = fk.segment_params(
                    length=length,
                    out_radius=params['segment_radii'][i],
                    wall_thickness=params['segment_thickness'][i]
                )
                segments.append(seg)
            
            # Store segments
            self.segments = segments
            
            # Compute material properties
            epsilon_pre = params['epsilon_pre']
            
            if params['material_model'] == "Neo-Hookean":
                mu = params['mu']
                model = fk.neohookean(mu=mu)
                etan_value = fk.tangent_modulus.neohookean(mu=model.mu, e=epsilon_pre)
            elif params['material_model'] == "Mooney-Rivlin":
                c1 = params['c1']
                c2 = params['c2']
                model = fk.mooney_rivlin(c1=c1, c2=c2)
                etan_value = fk.tangent_modulus.mooney_rivlin(c1=model.c1, c2=model.c2, e=epsilon_pre)
            elif params['material_model'] == "Ogden":
                mu = params['mu']
                alpha = params['alpha']
                model = fk.ogden(mu_input=mu, alpha=alpha)
                etan_value = fk.tangent_modulus.ogden(mu=model.mu, alpha=model.alpha, e=epsilon_pre)
            
            # Initialize GPU context
            resolution = self.resolution_spin.value()
            self.gpu_context.initialize(segments, resolution, epsilon_pre, etan_value)
            
            # Store for use in updates
            self.live_preview_params = params
            
            # Compute initial geometry with CURRENT pressures
            initial_points = self.compute_current_geometry()
            
            # Initialize visualization meshes with ACTUAL geometry
            self.initialize_live_meshes(initial_points)
            
            self.live_preview_active = True
            
            self.output_text.append("✓ Live preview enabled!")
            self.live_preview_checkbox.setText("🟢 Live Preview (GPU)")
            
        except RuntimeError as e:
            error_msg = str(e)
            self.output_text.append(f"✗ {error_msg}")
            self.live_preview_checkbox.setChecked(False)
            
        except Exception as e:
            self.output_text.append(f"✗ Failed to enable live preview: {str(e)}")
            self.live_preview_checkbox.setChecked(False)
            import traceback
            traceback.print_exc()

    def compute_current_geometry(self):
        """Compute geometry with current pressure values"""
        # Get current pressures
        P_A = self.pressure_sliders['A'].value() * 1e3
        P_B = self.pressure_sliders['B'].value() * 1e3
        P_C = self.pressure_sliders['C'].value() * 1e3
        P_D = self.pressure_sliders['D'].value() * 1e3
        
        params = self.live_preview_params
        
        # Compute moments
        channel_area = fk.channel_area(params['channel_radius'])
        centroid_dist = fk.centroid_distance(params['channel_radius'], params['septum_thickness'])
        
        M_ac = fk.directional_moment(P_A, P_C, channel_area, centroid_dist)
        M_bd = fk.directional_moment(P_B, P_D, channel_area, centroid_dist)
        M_res = fk.resultant_moment(M_ac, M_bd)
        phi = fk.bending_plane_ang(M_ac, M_bd)
        
        # Compute curvatures
        kappa = np.zeros(len(self.segments), dtype=np.float32)
        theta = np.zeros(len(self.segments), dtype=np.float32)
        phi_array = np.full(len(self.segments), phi, dtype=np.float32)
        
        epsilon_pre = params['epsilon_pre']
        
        for i, seg in enumerate(self.segments):
            EI = self.gpu_context.segment_EI[i]
            kappa[i] = fk.curvature(M_res, EI)
            theta[i] = fk.arc_angle(kappa[i], fk.prestrained_length(seg.length, epsilon_pre))
        
        # Update transforms
        self.gpu_context.update_transforms(kappa, theta, phi_array)
        
        # GPU update to get points
        points = self.gpu_context.update_fast(kappa, theta, phi_array)
        
        return points
                    
    def disable_live_preview(self):
        """Disable live preview and clean up GPU"""
        if self.live_preview_active:
            self.gpu_context.destroy()
            self.live_preview_active = False
            
            # Clear live mesh references
            self.live_meshes = {
                'centerline': None,
                'boundaries': None,
                'tubes': []
            }
            
            self.output_text.append("✓ Live preview disabled")
            self.live_preview_checkbox.setText("🔴 Live Preview (GPU)")
    
    def initialize_live_meshes(self, points=None):
        """Create PyVista meshes once for live preview (called only at start)"""
        # Clear plotter
        self.plotter.clear()
        self.plotter.set_background('white')
        
        # Add grid (static)
        if self.show_grid.isChecked():
            extent = 1.0
            spacing = 0.05
            coords = np.arange(-extent, extent + spacing, spacing)
            grid_xy = pv.RectilinearGrid(coords, coords, [0.0])
            self.plotter.add_mesh(grid_xy, color="black", style="wireframe", opacity=1.0)
        
        # If points are provided, use them; otherwise create dummy points
        if points is None:
            # Create initial straight line as dummy points
            resolution = self.resolution_spin.value()
            total_length = sum(seg.length for seg in self.segments)
            dummy_z = np.linspace(0, total_length, len(self.segments) * resolution)
            points = np.column_stack([
                np.zeros_like(dummy_z),
                np.zeros_like(dummy_z),
                dummy_z
            ])
        
        # Create centerline mesh
        centerline = pv.Spline(points, len(points))
        self.live_meshes['centerline'] = self.plotter.add_mesh(
            centerline,
            color='blue',
            line_width=3
        )
        
        # Create boundary points mesh
        if self.show_boundaries.isChecked():
            resolution = self.resolution_spin.value()
            segment_boundaries = np.arange(0, len(points), resolution)
            boundary_points = points[segment_boundaries]
            point_cloud = pv.PolyData(boundary_points)
            self.live_meshes['boundaries'] = self.plotter.add_mesh(
                point_cloud,
                color='red',
                point_size=15,
                render_points_as_spheres=True
            )
        
        # Create tube meshes for each segment
        palette_name = self.color_palette.currentText()
        colors = self.COLOR_PALETTES[palette_name]
        opacity = self.tube_opacity.value()
        
        self.live_meshes['tubes'].clear()
        
        # Create one tube per segment
        resolution = self.resolution_spin.value()
        points_per_segment = resolution
        for i, seg in enumerate(self.segments):
            start_idx = i * points_per_segment
            end_idx = (i + 1) * points_per_segment
            segment_points = points[start_idx:end_idx]
            
            # Skip if not enough points for a spline
            if len(segment_points) < 2:
                continue
                
            spline = pv.Spline(segment_points, 50)
            tube = spline.tube(radius=seg.out_radius, n_sides=12)
            tube_actor = self.plotter.add_mesh(
                tube,
                color=colors[i % len(colors)],
                opacity=opacity
            )
            self.live_meshes['tubes'].append(tube_actor)
        
        # Add coordinate frame (static)
        scale = 0.02
        position = np.array([0, 0, 0])
        
        arrow_x = pv.Arrow(start=position, direction=[1, 0, 0], scale=scale)
        self.plotter.add_mesh(arrow_x, color='red')
        
        arrow_y = pv.Arrow(start=position, direction=[0, 1, 0], scale=scale)
        self.plotter.add_mesh(arrow_y, color='green')
        
        arrow_z = pv.Arrow(start=position, direction=[0, 0, 1], scale=scale)
        self.plotter.add_mesh(arrow_z, color='blue')
        
        self.plotter.reset_camera()
        self.plotter.enable_parallel_projection()
        self.plotter.camera.parallel_scale = 0.2


    def on_pressure_changed_realtime(self, value):
        """Called every time a pressure slider moves"""
        if not self.live_preview_active:
            return
        
        # Debounce: use QTimer to avoid updating too frequently
        if self.update_timer is not None:
            self.update_timer.stop()
        
        from PySide6.QtCore import QTimer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_visualization_fast)
        self.update_timer.start(4)  # ~250 FPS max
    
    def update_visualization_fast(self):
        """Ultra-fast visualization update using GPU context (updates points only)"""
        if not self.live_preview_active:
            return
        
        try:
            import time
            start = time.time()
            
            # Get current pressures
            P_A = self.pressure_sliders['A'].value() * 1e3
            P_B = self.pressure_sliders['B'].value() * 1e3
            P_C = self.pressure_sliders['C'].value() * 1e3
            P_D = self.pressure_sliders['D'].value() * 1e3
            
            params = self.live_preview_params
            
            # Compute moments
            channel_area = fk.channel_area(params['channel_radius'])
            centroid_dist = fk.centroid_distance(params['channel_radius'], params['septum_thickness'])
            
            M_ac = fk.directional_moment(P_A, P_C, channel_area, centroid_dist)
            M_bd = fk.directional_moment(P_B, P_D, channel_area, centroid_dist)
            M_res = fk.resultant_moment(M_ac, M_bd)
            phi = fk.bending_plane_ang(M_ac, M_bd)
            
            # Compute curvatures
            kappa = np.zeros(len(self.segments), dtype=np.float32)
            theta = np.zeros(len(self.segments), dtype=np.float32)
            phi_array = np.full(len(self.segments), phi, dtype=np.float32)
            
            epsilon_pre = params['epsilon_pre']
            
            for i, seg in enumerate(self.segments):
                EI = self.gpu_context.segment_EI[i]
                kappa[i] = fk.curvature(M_res, EI)
                theta[i] = fk.arc_angle(kappa[i], fk.prestrained_length(seg.length, epsilon_pre))
            
            # Update transforms
            self.gpu_context.update_transforms(kappa, theta, phi_array)
            
            # GPU update (FAST!)
            points = self.gpu_context.update_fast(kappa, theta, phi_array)
            self.points = points
            
            # NEW: Update meshes by replacing them (PyVista approach)
            resolution = self.resolution_spin.value()
            segment_boundaries = np.arange(0, len(points), resolution)
            
            # Update centerline
            centerline = pv.Spline(points, resolution)
            self.live_meshes['centerline'].mapper.SetInputData(centerline)
            
            # Update boundaries
            if self.live_meshes['boundaries'] is not None:
                boundary_points = points[segment_boundaries]
                point_cloud = pv.PolyData(boundary_points)
                self.live_meshes['boundaries'].mapper.SetInputData(point_cloud)
            
            # Update tubes
            for i, tube_actor in enumerate(self.live_meshes['tubes']):
                start_idx = segment_boundaries[i]
                end_idx = segment_boundaries[i+1] if i < len(self.segments)-1 else len(points)
                segment_points = points[start_idx:end_idx]
                
                spline = pv.Spline(segment_points, 50)
                tube = spline.tube(radius=self.segments[i].out_radius, n_sides=12)
                tube_actor.mapper.SetInputData(tube)
            
            # Force render update
            self.plotter.render()
            
            # Show FPS
            elapsed = (time.time() - start) * 1000
            fps = 1000.0 / elapsed if elapsed > 0 else 0
            self.live_fps_label.setText(f"FPS: {fps:.1f} ({elapsed:.1f}ms)")
            
        except Exception as e:
            print(f"Live preview error: {e}")
            import traceback
            traceback.print_exc()
        
        def closeEvent(self, event):
            """Clean up GPU resources on exit"""
            if self.gpu_context.initialized:
                self.gpu_context.destroy()
            event.accept()

@timeit
def compute_simulation_fk(params):
    """Forward Kinematics: Pressures → End-effector pose"""
    info_lines = []
    
    # Extract parameters
    channel_a = params['channel_a']
    channel_b = params['channel_b']
    channel_c = params['channel_c']
    channel_d = params['channel_d']
    channel_radius = params['channel_radius']
    septum_thickness = params['septum_thickness']
    epsilon_pre = params['epsilon_pre']
    bulk_modulus = params['bulk_modulus']
    material_model = params['material_model']
    resolution = params.get('resolution', 1000)

    segments = []
    for i, length in enumerate(params['segment_lengths']):
        seg = fk.segment_params(
            length=length,
            out_radius=params['segment_radii'][i],
            wall_thickness=params['segment_thickness'][i]
        )
        print(f"DEBUG: Segment {i+1}: length={length}, out_radius={seg.out_radius}, wall_thickness={seg.wall_thickness}")
        segments.append(seg)
    
    for i, seg in enumerate(segments):
        if not checks.test_condition_4(seg.out_radius, seg.wall_thickness):
            ratio = seg.wall_thickness / seg.out_radius
            raise ValueError(
                f"!! Segment {i+1}: Thin-wall assumption violated!\n"
                f"   Wall thickness/Radius ratio: {ratio:.3f} (must be < 0.2)\n"
                f"   Outer radius: {seg.out_radius*1000:.2f} mm\n"
                f"   Wall thickness: {seg.wall_thickness*1000:.2f} mm"
            )

    # Material model
    try : 
        if params['material_model'] == "Neo-Hookean":
            mu = params['mu']
            model = fk.neohookean(mu=mu)
            etan_value = fk.tangent_modulus.neohookean(mu=model.mu, e=epsilon_pre)
        elif params['material_model'] == "Mooney-Rivlin":
            c1 = params['c1']
            c2 = params['c2']
            model = fk.mooney_rivlin(c1=c1, c2=c2)
            etan_value = fk.tangent_modulus.mooney_rivlin(c1=model.c1, c2=model.c2, e=epsilon_pre)
        elif params['material_model'] == "Ogden":
            mu = params['mu']
            alpha = params['alpha']
            model = fk.ogden(mu_input=mu, alpha=alpha)
            etan_value = fk.tangent_modulus.ogden(mu=model.mu, alpha=model.alpha, e=epsilon_pre)
        else:
            raise ValueError(f"!! Unsupported material model: {material_model}")
    except KeyError as e:
        raise ValueError(f"!! Missing material parameter: {e}")

    if not checks.test_condition_1(mu, bulk_modulus):
        E_approx = 3 * mu
        E_full = 9 * bulk_modulus * mu / (3 * bulk_modulus + mu)
        percent_diff = abs(E_approx - E_full) / E_full * 100
        raise ValueError(
            f"!! Material incompressibility assumption violated!\n"
            f"   E_approx = 3μ = {E_approx:.2e} Pa\n"
            f"   E_full = 9Kμ/(3K+μ) = {E_full:.2e} Pa\n"
            f"   Percent difference: {percent_diff:.2f}% (must be < 2%)\n"
            f"   Suggestion: Increase bulk modulus K or adjust μ"
        )

    # Compute rigidities
    for i, seg in enumerate(segments):
        seg.I = fk.compute_second_moment(seg.out_radius, seg.wall_thickness)
        seg.EI = fk.flexural_rigidity(etan_value, seg.I)
        info_lines.append(f"Segment {i+1}: EI = {seg.EI:.4e} Nm²")
    
    # Actuation moments
    channel_area = fk.channel_area(channel_radius)
    centroid_dist = fk.centroid_distance(channel_radius, septum_thickness)
    M_ac = fk.directional_moment(channel_a, channel_c, channel_area, centroid_dist)
    M_bd = fk.directional_moment(channel_b, channel_d, channel_area, centroid_dist)
    M_res = fk.resultant_moment(M_ac, M_bd)
    phi = fk.bending_plane_ang(M_ac, M_bd)

    info_lines.append(f"\nResultant Moment: {M_res:.4e} Nm")
    info_lines.append(f"Bending Plane: {math.degrees(phi):.2f}°")
    
    # Compute curvatures
    for i, seg in enumerate(segments):
        seg.curvature = fk.curvature(M_res, seg.EI)
        seg.theta = fk.arc_angle(seg.curvature, fk.prestrained_length(seg.length, epsilon_pre))
        print(seg.curvature, seg.theta)
        info_lines.append(f"Segment {i+1}: κ={seg.curvature:.4e} 1/m, θ={math.degrees(seg.theta):.2f}°")
    
    for i, kappa in enumerate(seg.curvature for seg in segments):
        if not checks.test_condition_2(kappa):
            raise ValueError(
                f"!! Segment {i+1}: Curvature out of expected range!\n"
                f"   Curvature: {kappa:.2f} rad/m\n"
                f"   Expected range: 0-100 rad/m\n"
                f"   • κ > 100: Risk of buckling or material failure\n"
            )

    theta = [seg.theta for seg in segments]
    passed, message = checks.test_condition_3_geometric(segments, theta, phi, epsilon_pre)
    
    if not passed:
        raise ValueError(f"{message}")
    else:
        info_lines.append(f"\n✓ Collision Check: {message}")

    # Generate spline
    kappa = np.array([seg.curvature for seg in segments], dtype=np.float32)
    theta = np.array([seg.theta for seg in segments], dtype=np.float32)
    phi_array = np.array([phi] * len(segments), dtype=np.float32)
    length = np.array([seg.length for seg in segments], dtype=np.float32)

    T_cumulative = spline.compute_cumulative_transforms(kappa, theta, phi_array, len(segments))
    points = generate_robot_spline(kappa, theta, phi_array, length, T_cumulative, resolution, len(segments))
    
    return points, segments, '\n'.join(info_lines)


def compute_simulation_ik(params):
    """
    Inverse Kinematics: Target pose → Pressures using CMA-ES + DLS
    """    
    info_lines = []
    info_lines.append("=== INVERSE KINEMATICS ===\n")
    
    # Extract target
    target_pos = np.array([
        params['target_x'],
        params['target_y'],
        params['target_z']
    ])
    
    info_lines.append(f"Target position: [{target_pos[0]*1000:.1f}, {target_pos[1]*1000:.1f}, {target_pos[2]*1000:.1f}] mm\n")
    
    # Create segments and compute material properties
    segments = []
    for i, length in enumerate(params['segment_lengths']):
        seg = fk.segment_params(
            length=length,
            out_radius=params['segment_radii'][i],
            wall_thickness=params['segment_thickness'][i]
        )
        segments.append(seg)
    
    # Compute tangent modulus and rigidities
    epsilon_pre = params['epsilon_pre']
    
    if params['material_model'] == "Neo-Hookean":
        mu = params['mu']
        model = fk.neohookean(mu=mu)
        etan_value = fk.tangent_modulus.neohookean(mu=model.mu, e=epsilon_pre)
    elif params['material_model'] == "Mooney-Rivlin":
        c1 = params['c1']
        c2 = params['c2']
        model = fk.mooney_rivlin(c1=c1, c2=c2)
        etan_value = fk.tangent_modulus.mooney_rivlin(c1=model.c1, c2=model.c2, e=epsilon_pre)
    elif params['material_model'] == "Ogden":
        mu = params['mu']
        alpha = params['alpha']
        model = fk.ogden(mu_input=mu, alpha=alpha)
        etan_value = fk.tangent_modulus.ogden(mu=model.mu, alpha=model.alpha, e=epsilon_pre)
    
    for i, seg in enumerate(segments):
        seg.I = fk.compute_second_moment(seg.out_radius, seg.wall_thickness)
        seg.EI = fk.flexural_rigidity(etan_value, seg.I)
        info_lines.append(f"Segment {i+1}: EI = {seg.EI:.4e} Nm²")
    
    # Pressure bounds
    pressure_bounds = np.array([
        [0, 100e3],  # pA: 0-100 kPa
        [0, 100e3],  # pB
        [0, 100e3],  # pC
        [0, 100e3]   # pD
    ])
    
    # Run hybrid IK solver
    try:
        solution_pressures, final_error, info_dict = ik.hybrid_cmaes_gradient_solver(
            target_pos,
            params,
            segments,
            pressure_bounds,
            cma_iterations=params.get('ik_max_iter', 50),
            cma_popsize=15,
            gd_tolerance=params['ik_tolerance']
        )
        
        # Now run FK with solution pressures to get visualization
        fk_params = params.copy()
        fk_params['channel_a'] = solution_pressures[0]
        fk_params['channel_b'] = solution_pressures[1]
        fk_params['channel_c'] = solution_pressures[2]
        fk_params['channel_d'] = solution_pressures[3]
        
        points, segments, fk_info = compute_simulation_fk(fk_params)
        
        # Build solution data
        solution_data = {
            'pressures': {
                'A': solution_pressures[0],
                'B': solution_pressures[1],
                'C': solution_pressures[2],
                'D': solution_pressures[3]
            },
            'converged': info_dict['converged'],
            'iterations': info_dict['cma_iterations'] + info_dict['dls_iterations'],
            'final_error': final_error * 1000,  # Convert to mm
            'cma_iterations': info_dict['cma_iterations'],
            'dls_iterations': info_dict['dls_iterations'],
            'total_time': info_dict['cma_time'] + info_dict['dls_time']
        }
        
        info_lines.append("\n" + fk_info)
        
        return points, segments, '\n'.join(info_lines), solution_data
        
    except Exception as e:
        import traceback
        error_msg = f"IK Solver Error:\n{str(e)}\n\n{traceback.format_exc()}"
        raise ValueError(error_msg)


def generate_robot_spline(kappa, theta, phi, length, T_cumulative, resolution, num_segments):
    """Generate spline using CUDA"""
    kappa = np.ascontiguousarray(kappa, dtype=np.float32)
    theta = np.ascontiguousarray(theta, dtype=np.float32)
    phi = np.ascontiguousarray(phi, dtype=np.float32)
    length = np.ascontiguousarray(length, dtype=np.float32)
    T_flat = np.ascontiguousarray(T_cumulative.reshape(num_segments, 16), dtype=np.float32)
    
    total_points = num_segments * resolution
    output = np.zeros(total_points * 3, dtype=np.float32)
    
    error_code = lib.generate_spline_points(
        kappa, theta, phi, length, T_flat, resolution, num_segments, output
    )
    
    if error_code != 0:
        raise RuntimeError(f"CUDA error code: {error_code}")
    
    points = output.reshape(-1, 3)      # Reshape to (N, 3) format
    print(points[-1:])                  # Debug: print last point
    return points

