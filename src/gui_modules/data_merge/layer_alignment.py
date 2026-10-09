import os
import customtkinter as ctk
from tkinter import filedialog
import numpy as np
import pprint

from core_backend.data_merge.spatial_reference import GlobalSpatialReference
from core_backend.data_merge.hdf5_master_exporter import MasterHDF5Exporter
from gui_modules.data_merge.interactive_alignment import InteractiveAlignmentTool
from core_backend.data_merge.logger_config import PipelineLogger

logger = PipelineLogger()

class InputSelectionDialog(ctk.CTkToplevel):
    """
    Structured popup dialog for selecting global JSONs and layer files.
    Enforces a strict Top-to-Bottom sample order to prevent user input errors.
    """
    def __init__(self, master, callback):
        super().__init__(master)
        self.title("Select Structured Input Data for Alignment")
        self.geometry("950x850")
        self.callback = callback
        
        # Make the dialog modal
        self.transient(master)
        self.grab_set()

        # Create main scrollable area
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- 1. Global Reference Files ---
        global_frame = ctk.CTkFrame(self.scroll_frame, fg_color="#1B2433")
        global_frame.pack(fill="x", pady=(0, 15), padx=5)
        
        ctk.CTkLabel(global_frame, text="Global Reference Files (JSON)", font=("Arial", 16, "bold"), text_color="#E74C3C").pack(anchor="w", padx=10, pady=5)
        
        # Analysis Results
        frame_analysis = ctk.CTkFrame(global_frame, fg_color="transparent")
        frame_analysis.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(frame_analysis, text="Hydrate Data (analysis_results.json):", width=250, anchor="w").pack(side="left")
        self.analysis_path = ctk.CTkEntry(frame_analysis)
        self.analysis_path.pack(side="left", fill="x", expand=True, padx=10)
        ctk.CTkButton(frame_analysis, text="Browse", width=80, command=lambda: self.browse_file(self.analysis_path, [("JSON Files", "*.json")])).pack(side="right")
        
        # Metadata
        frame_meta = ctk.CTkFrame(global_frame, fg_color="transparent")
        frame_meta.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(frame_meta, text="Metadata (metadata_preprocessed.json):", width=250, anchor="w").pack(side="left")
        self.meta_path = ctk.CTkEntry(frame_meta)
        self.meta_path.pack(side="left", fill="x", expand=True, padx=10)
        ctk.CTkButton(frame_meta, text="Browse", width=80, command=lambda: self.browse_file(self.meta_path, [("JSON Files", "*.json")])).pack(side="right")

        # --- 2. Structured Layer Sequence ---
        # Definierte physikalische Reihenfolge des Eisbohrkerns (Top -> Bottom)
        sequence = [
            {"id": "Top", "label": "1. Top Surface (Microscope)", "img": True, "h5": False, "mirror": False},
            {"id": "FA1", "label": "2. Fabric Analyzer 1 (FA1)", "img": False, "h5": True, "mirror": True},
            {"id": "Layer 1", "label": "3. Layer 1 (Microscope)", "img": True, "h5": True, "mirror": True},
            {"id": "Layer 2", "label": "4. Layer 2 (Microscope)", "img": True, "h5": True, "mirror": False},
            {"id": "FA2", "label": "5. Fabric Analyzer 2 (FA2)", "img": False, "h5": True, "mirror": True},
            {"id": "Layer 3", "label": "6. Layer 3 (Microscope)", "img": True, "h5": True, "mirror": True},
            {"id": "Layer 4", "label": "7. Layer 4 (Microscope)", "img": True, "h5": True, "mirror": False},
            {"id": "FA3", "label": "8. Fabric Analyzer 3 (FA3)", "img": False, "h5": True, "mirror": False},
            {"id": "Bottom", "label": "9. Bottom Surface (Microscope)", "img": True, "h5": False, "mirror": False}
        ]

        self.level_entries = {}

        # Create one input block for each layer in the configured sequence.
        for item in sequence:
            frame = ctk.CTkFrame(self.scroll_frame, fg_color="#212F3D")
            frame.pack(fill="x", pady=5, padx=5)

            # Indicate whether the layer will be mirrored during processing.
            mirror_txt = "(Auto-Mirrored on X-Axis)" if item["mirror"] else ""
            header = ctk.CTkLabel(frame, text=f"{item['label']}  {mirror_txt}", font=("Arial", 15, "bold"), text_color="#F1C40F")
            header.grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=5)

            row_idx = 1
            img_entry = None
            h5_entry = None

            # Image input for microscope layers.
            if item["img"]:
                ctk.CTkLabel(frame, text="Image (.tif):", width=140, anchor="e").grid(row=row_idx, column=0, padx=5, pady=2)
                img_entry = ctk.CTkEntry(frame, width=450)
                img_entry.grid(row=row_idx, column=1, padx=5, pady=2)
                ctk.CTkButton(frame, text="Browse", width=60, command=lambda e=img_entry: self.browse_file(e, [("Image Files", "*.tif *.tiff *.png")])).grid(row=row_idx, column=2, padx=5, pady=2)
                row_idx += 1

            # HDF5 input for fabric-analysis or boundary data.
            if item["h5"]:
                label_txt = "FA Data (.h5):" if not item["img"] else "Boundaries (.h5):"
                ctk.CTkLabel(frame, text=label_txt, width=140, anchor="e").grid(row=row_idx, column=0, padx=5, pady=2)
                h5_entry = ctk.CTkEntry(frame, width=450)
                h5_entry.grid(row=row_idx, column=1, padx=5, pady=2)
                ctk.CTkButton(frame, text="Browse", width=60, command=lambda e=h5_entry: self.browse_file(e, [("HDF5 Files", "*.h5")])).grid(row=row_idx, column=2, padx=5, pady=2)
                row_idx += 1

            # Gemeinsames Z-Abstandsfeld pro Ebene
            ctk.CTkLabel(frame, text="Z-Distance (mm):", width=140, anchor="e").grid(row=row_idx, column=0, padx=5, pady=(2, 10))
            z_entry = ctk.CTkEntry(frame, width=100)
            z_entry.insert(0, "0.0")
            z_entry.grid(row=row_idx, column=1, sticky="w", padx=5, pady=(2, 10))

            self.level_entries[item["id"]] = {
                "img_entry": img_entry,
                "h5_entry": h5_entry,
                "z_entry": z_entry,
                "mirror": item["mirror"]
            }

        # --- 3. Start Button (Fixed at bottom) ---
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(fill="x", padx=10, pady=10)
        self.start_btn = ctk.CTkButton(self.btn_frame, text="Confirm & Initialize Pipeline", height=40, 
                                       fg_color="#27AE60", hover_color="#1E8449", font=("Arial", 16, "bold"), 
                                       command=self.compile_and_emit)
        self.start_btn.pack(fill="x")

    def browse_file(self, entry_widget, filetypes):
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path:
            entry_widget.delete(0, 'end')
            entry_widget.insert(0, file_path)

    def compile_and_emit(self):
        """Extracts data from the rigid layout and formats it for the backend."""
        config = {
            "global_references": {
                "analysis_results": self.analysis_path.get(),
                "metadata_preprocessed": self.meta_path.get()
            },
            "layers": []
        }
        
        for layer_id, widgets in self.level_entries.items():
            img_path = widgets["img_entry"].get() if widgets["img_entry"] else ""
            h5_path = widgets["h5_entry"].get() if widgets["h5_entry"] else ""

            # Ignore layer entries that do not specify an image or HDF5 file.
            if not img_path and not h5_path:
                continue 

            try:
                z_val = float(widgets["z_entry"].get())
            except ValueError:
                z_val = 0.0
                
            # Use the image for microscope layers and the HDF5 file for fabric-analysis layers.
            primary_file = img_path if img_path else h5_path
            
            layer_data = {
                "type": layer_id,
                "file_path": primary_file,
                "z_distance_mm": z_val,
                "mirror": widgets["mirror"]
            }
            
            # Preserve the associated HDF5 analysis file for microscope layers.
            if img_path and h5_path:
                layer_data["extra_h5_path"] = h5_path

            config["layers"].append(layer_data)
            
        self.callback(config)
        self.destroy()


class LayerAlignmentFrame(ctk.CTkFrame):
    def __init__(self, master, controller):
        super().__init__(master, fg_color="transparent")
        self.controller = controller
        
        self.config_data = {}
        self.layer_map = {}
        self.alignment_queue = []
        self.current_alignment_tool = None
        
        self.setup_ui()

    def setup_ui(self):
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.place(relx=0.5, rely=0.5, anchor="center")

        title = ctk.CTkLabel(self.container, text="Layer Alignment & Master HDF5 Export", font=("Arial", 28, "bold"))
        title.pack(pady=20)

        self.select_input_btn = ctk.CTkButton(self.container, text="Start New Alignment", font=("Arial", 16, "bold"), 
                                              height=50, width=250, command=self.open_input_dialog)
        self.select_input_btn.pack(pady=10)

        self.load_state_btn = ctk.CTkButton(self.container, text="Load Saved Alignment State", font=("Arial", 16, "bold"), 
                                              height=50, width=250, fg_color="#8E44AD", hover_color="#732D91", command=self.load_state)
        self.load_state_btn.pack(pady=10)

        self.status_label = ctk.CTkLabel(self.container, text="Status: Awaiting input...", font=("Arial", 14), text_color="gray")
        self.status_label.pack(pady=10)

    def open_input_dialog(self):
        dialog = InputSelectionDialog(self, callback=self.handle_alignment_configuration)

    def load_state(self):
        """Liest eine zuvor gespeicherte JSON-Alignment-Datei ein und springt direkt zum Export."""
        import json
        state_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")], title="Load Alignment State")
        if not state_path: return
        
        try:
            with open(state_path, 'r') as f:
                loaded_config = json.load(f)
            
            self.config_data = loaded_config
            self.layer_map = {}
            
            # Restore serialized transform matrices as NumPy arrays.
            for layer in self.config_data.get("layers", []):
                if "transform_matrix" in layer:
                    layer["transform_matrix"] = np.array(layer["transform_matrix"], dtype=np.float64)
                self.layer_map[layer["type"]] = layer
                
            global_refs = self.config_data.get("global_references", {})
            self.spatial_ref = GlobalSpatialReference(global_refs.get("analysis_results"), global_refs.get("metadata_preprocessed"))
            
            self.status_label.configure(text="Status: State loaded successfully. Ready for HDF5 export.", text_color="#2ECC71")
            logger.info("Alignment State successfully loaded. Skipping manual alignment.")
            
            # Resume directly at export because this alignment state is already saved.
            self.execute_hdf5_export(skip_save=True)
            
        except Exception as e:
            self.status_label.configure(text=f"Error loading state: {str(e)}", text_color="red")
            logger.error(f"Load state failed: {e}")

    def handle_alignment_configuration(self, config_data):
        self.config_data = config_data
        self.status_label.configure(text="Status: Preparing Alignment Phases...", text_color="yellow")
        
        global_refs = config_data.get("global_references", {})
        try:
            self.spatial_ref = GlobalSpatialReference(global_refs.get("analysis_results"), global_refs.get("metadata_preprocessed"))
        except Exception as e:
            self.status_label.configure(text=f"Error parsing JSONs: {str(e)}", text_color="red")
            return

        self.layer_map = {layer["type"]: layer for layer in config_data.get("layers", [])}
        self.alignment_queue = []
        
        if "Top" in self.layer_map:
            if "Layer 1" in self.layer_map: self.alignment_queue.append(("Phase 1: Surface Alignment", "Top", "Layer 1", False))
            if "FA1" in self.layer_map:     self.alignment_queue.append(("Phase 1: Surface Alignment", "Top", "FA1", True))
            
        if "Bottom" in self.layer_map:
            if "Layer 2" in self.layer_map: self.alignment_queue.append(("Phase 2: Bottom Alignment", "Bottom", "Layer 2", False))
            if "Layer 4" in self.layer_map: self.alignment_queue.append(("Phase 2: Bottom Alignment", "Bottom", "Layer 4", False))
            if "FA3" in self.layer_map:     self.alignment_queue.append(("Phase 2: Bottom Alignment", "Bottom", "FA3", True))
            
        if "Layer 2" in self.layer_map:
            if "Layer 3" in self.layer_map: self.alignment_queue.append(("Phase 3: Middle Gap Bridge", "Layer 2", "Layer 3", False))
            if "FA2" in self.layer_map:     self.alignment_queue.append(("Phase 3: Middle Gap Bridge", "Layer 2", "FA2", True))

        logger.info(f"Generated Phase Queue with {len(self.alignment_queue)} alignment tasks.")
        self.run_next_alignment()

    def run_next_alignment(self):
        if self.current_alignment_tool is not None:
            self.current_alignment_tool.destroy()
            self.current_alignment_tool = None

        if not self.alignment_queue:
            logger.info("All alignments completed. Moving to HDF5 Export.")
            self.container.place(relx=0.5, rely=0.5, anchor="center") 
            self.execute_hdf5_export(skip_save=False)
            return

        phase_name, anchor_type, target_type, allow_scaling = self.alignment_queue.pop(0)
        anchor_path = self.layer_map[anchor_type]["file_path"]
        target_path = self.layer_map[target_type]["file_path"]

        print(f"\n[DEBUG] Lade ANCHOR Bild exakt von: {anchor_path}")
        print(f"[DEBUG] Lade TARGET Bild exakt von: {target_path}\n")
        
        is_anchor_fa = "FA" in anchor_type
        is_target_fa = "FA" in target_type
        anchor_mirror = self.layer_map[anchor_type].get("mirror", False)
        target_mirror = self.layer_map[target_type].get("mirror", False)
        
        self.status_label.configure(text=f"Status: {phase_name} - Aligning {target_type} to {anchor_type}...", text_color="#3498DB")
        logger.info(f"Opening Tool: Phase=[{phase_name}], Anchor=[{anchor_type}], Target=[{target_type}], Scaling=[{allow_scaling}]")

        self.container.place_forget()
        self.current_target_type = target_type
        
        self.current_alignment_tool = InteractiveAlignmentTool(
            self, anchor_path=anchor_path, target_path=target_path, allow_scaling=allow_scaling,
            is_anchor_fa=is_anchor_fa, is_target_fa=is_target_fa, anchor_mirror=anchor_mirror,
            target_mirror=target_mirror, phase_name=phase_name, anchor_name=anchor_type,      
            target_name=target_type, callback=self.on_pair_aligned
        )
        self.current_alignment_tool.pack(fill="both", expand=True)

    def on_pair_aligned(self, calculated_matrix):
        logger.info(f"Matrix accepted for {self.current_target_type}.")
        self.layer_map[self.current_target_type]["transform_matrix"] = calculated_matrix
        self.run_next_alignment()

    def execute_hdf5_export(self, skip_save=False):
        # Offer to save the alignment state before exporting a new result.
        if not skip_save:
            state_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json")],
                title="Save Alignment State (Cancel to skip)"
            )
            if state_path:
                try:
                    import json
                    save_config = {
                        "global_references": self.config_data.get("global_references", {}),
                        "layers": []
                    }
                    # Convert matrices to lists so the configuration can be serialized as JSON.
                    for layer in self.config_data.get("layers", []):
                        l_copy = layer.copy()
                        mat = self.layer_map[layer["type"]].get("transform_matrix", np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
                        l_copy["transform_matrix"] = mat.tolist() 
                        save_config["layers"].append(l_copy)
                        
                    with open(state_path, 'w') as f:
                        json.dump(save_config, f, indent=4)
                    self.status_label.configure(text=f"Status: State saved to {os.path.basename(state_path)}", text_color="#2ECC71")
                    logger.info(f"Saved alignment state to {state_path}")
                except Exception as e:
                    logger.error(f"Failed to save state: {e}")

        # Select the destination for the master HDF5 file.
        output_path = filedialog.asksaveasfilename(
            defaultextension=".h5",
            filetypes=[("HDF5 Files", "*.h5")],
            title="Save Master HDF5 File As..."
        )
        
        if not output_path:
            self.status_label.configure(text="Status: Export cancelled.", text_color="gray")
            return

        self.status_label.configure(text="Status: Building Master HDF5... (This may take a moment)", text_color="yellow")
        self.update_idletasks()
        
        try:
            # Compose each layer's local transform with its anchor transform.
            
            # Convert OpenCV affine matrices to homogeneous 3x3 form for composition.
            def get_3x3(m):
                M3 = np.eye(3)
                if m.shape == (2, 3):
                    M3[:2, :] = m
                elif m.shape == (3, 3):
                    M3 = m
                return M3

            global_matrices = {}
            
            # Top and bottom layers define the global reference frame.
            global_matrices["Top"] = np.eye(3)
            global_matrices["Bottom"] = np.eye(3)

            # First-level layers are aligned directly to a global anchor.
            for layer_type in ["Layer 1", "FA1", "Layer 2", "Layer 4", "FA3"]:
                if layer_type in self.layer_map:
                    rel_mat = self.layer_map[layer_type].get("transform_matrix", np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
                    global_matrices[layer_type] = get_3x3(rel_mat)

            # Second-level layers inherit the transform of Layer 2.
            if "Layer 2" in global_matrices:
                m_layer2_global = global_matrices["Layer 2"]
                
                for sub_layer in ["Layer 3", "FA2"]:
                    if sub_layer in self.layer_map:
                        m_rel = get_3x3(self.layer_map[sub_layer].get("transform_matrix", np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])))
                        # Compose the anchor's global transform with the target's relative transform.
                        global_matrices[sub_layer] = m_layer2_global @ m_rel

            # Pass the final affine transforms to the exporter.
            for layer in self.config_data["layers"]:
                l_type = layer["type"]
                if l_type in global_matrices:
                    layer["transform_matrix"] = global_matrices[l_type][:2, :]
                else:
                    layer["transform_matrix"] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

            # =================================================================

            exporter = MasterHDF5Exporter(output_path, self.spatial_ref, self.config_data)
            exporter.build_master_file()
            
            self.status_label.configure(text=f"Status: Success! Master HDF5 saved to:\n{os.path.basename(output_path)}", text_color="#2ECC71")
        except Exception as e:
            self.status_label.configure(text=f"Status Error: Export failed ({str(e)})", text_color="red")
            logger.error(f"HDF5 Export Failed: {e}")