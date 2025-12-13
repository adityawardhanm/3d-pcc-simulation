# materials.py
## 1.2 Common Parameters

# channel geometry parameters (Two semi-circular channels, separated by a septum wall)
channel_rad = 5.0e-3  # in meters
septum_thickness = 8.0e-4  # in meters

# actuation pressures (uniform across all segments)
channel_a_pressure = 6.0e4 # in Pa
channel_b_pressure = 0.0e4  # in Pa
channel_c_pressure = 0.0e4  # in Pa
channel_d_pressure = 0.0e4  # in Pa

# manufacturing pre-strains
epsilon_pre = 1.5e-1
lambda_pre = 1 + epsilon_pre

poisson_ratio = 5.0e-1  # dimensionless

## 1.3 Individual Segment Parameters




# def neohookean(mu, e):
#     class NeoHookean:
#         def __init__(self, mu):
#             self.mu = mu
#             self.e = e

#         @property
#         def etan(self):
#             lam = 1 + self.e
#             etan = self.mu * ((2 * lam + lam**-2))
#             return etan
        

# mod.py





            # # Color map for segments (extended palette)
            # colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'cyan', 'magenta', 'brown', 'pink']

            # # Draw each segment as a tube
            # for i in range(num_segments):
            #     start_idx = segment_boundaries[i]
            #     end_idx = segment_boundaries[i+1] if i < 4 else len(self.points)
                
            #     segment_points = self.points[start_idx:end_idx]
                
            #     # Create polyline for this segment
            #     poly = pv.PolyData(segment_points)
            #     poly.lines = np.hstack([[len(segment_points)] + list(range(len(segment_points)))])
                
            #     # Create tube with segment-specific radius
            #     tube = poly.tube(radius=self.segments[i].out_radius)
            #     self.plotter.add_mesh(
            #         tube,
            #         color=colors[i],
            #         opacity=0.8,
            #         smooth_shading=True,
            #         label=f"Segment {i+1}"
            #     )
            
            # # Add coordinate frame at base
            # self.plotter.add_axes_at_origin(
            #     labels_off=False,
            #     line_width=3,
            #     xlabel='X',
            #     ylabel='Y',
            #     zlabel='Z'
            # )
            
            # # Add text overlay with parameters
            # params = self.get_current_params()
            # info_text = (
            #     f"Pressures: A={params['channel_a']/1e2:.1f}Pa, "
            #     f"B={params['channel_b']/1e2:.1f}Pa, "
            #     f"C={params['channel_c']/1e2:.1f}Pa, "
            #     f"D={params['channel_d']/1e2:.1f}Pa\n"
            #     f"Total Length: {sum([s.length for s in self.segments])*1000:.1f}mm\n"
            #     f"Tip: ({self.points[-1, 0]*1000:.1f}, "
            #     f"{self.points[-1, 1]*1000:.1f}, "
            #     f"{self.points[-1, 2]*1000:.1f}) mm"
            # )
            # self.plotter.add_text(
            #     info_text,
            #     position='upper_left',
            #     font_size=10,
            #     color='black'
            #  )
            
            # # Reset camera to view entire robot
            # self.plotter.reset_camera()

from PySide6.QtWidgets import QScrollArea
 


# preview_group = QGroupBox("Color Preview")
# preview_layout = QVBoxLayout(preview_group)

# preview_grid = QGridLayout()

# # Centerline color

# preview_grid.addWidget(QLabel("Centerline:"), 5, 0)
# self.centerline_color_label = QLabel()
# self.centerline_color_label.setFixedSize(80, 25)
# self.centerline_color_label.setStyleSheet("background-color: blue; border: 1px solid black;")
# preview_grid.addWidget(self.centerline_color_label, 5, 1)

# preview_grid.addWidget(QLabel("Segment Ends:"), 6, 0)
# self.ends_color_label = QLabel()
# self.ends_color_label.setFixedSize(80, 25)
# self.ends_color_label.setStyleSheet("background-color: red; border: 1px solid black;")
# preview_grid.addWidget(self.ends_color_label, 6, 1)

# scroll = QScrollArea()
# scroll.setWidgetResizable(True)
# scroll.setMaximumHeight(300)

# self.segment_colors_widget = QWidget()
# self.segment_colors_layout = QVBoxLayout(self.segment_colors_widget)
# scroll.setWidget(self.segment_colors_widget)

# preview_layout.addLayout(scroll)

# preview_layout.addLayout(preview_grid)

# layout.addWidget(preview_group)

