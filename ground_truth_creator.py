#!/usr/bin/env python3
"""
Ground Truth Creator für Spritesheet Frame-Validierung
Ermöglicht manuelle Annotation korrekter Frame-Anzahlen für systematische Evaluation
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import random


class GroundTruthCreator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Spritesheet Ground Truth Creator")
        self.root.geometry("1200x800")

        self.current_image = None
        self.current_filename = None
        self.ground_truth_data = {}
        self.annotation_file = "spritesheet_ground_truth.json"

        # Lade existierende Ground Truth falls vorhanden
        self.load_existing_ground_truth()

        self.setup_ui()

    def setup_ui(self):
        # Main Frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control Panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(control_frame, text="Load Image",
                   command=self.load_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Random Sample",
                   command=self.load_random_sample).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Save Ground Truth",
                   command=self.save_ground_truth).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Export Report",
                   command=self.export_report).pack(side=tk.LEFT, padx=(0, 5))

        # Image Frame
        image_frame = ttk.LabelFrame(main_frame, text="Spritesheet Image")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(padx=10, pady=10)

        # Annotation Frame
        annotation_frame = ttk.LabelFrame(main_frame, text="Manual Annotation")
        annotation_frame.pack(fill=tk.X)

        # Frame Count Input
        frame_frame = ttk.Frame(annotation_frame)
        frame_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(frame_frame, text="Correct Frame Count:").pack(side=tk.LEFT)
        self.frame_count_var = tk.StringVar()
        self.frame_count_entry = ttk.Entry(
            frame_frame, textvariable=self.frame_count_var, width=10)
        self.frame_count_entry.pack(side=tk.LEFT, padx=(5, 10))

        # Quality Assessment
        quality_frame = ttk.Frame(annotation_frame)
        quality_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(quality_frame, text="Background Complexity:").pack(
            side=tk.LEFT)
        self.complexity_var = tk.StringVar(value="Medium")
        complexity_combo = ttk.Combobox(quality_frame, textvariable=self.complexity_var,
                                        values=["Simple", "Medium", "Complex"], width=10)
        complexity_combo.pack(side=tk.LEFT, padx=(5, 10))

        ttk.Label(quality_frame, text="Sprite Layout:").pack(side=tk.LEFT)
        self.layout_var = tk.StringVar(value="Grid")
        layout_combo = ttk.Combobox(quality_frame, textvariable=self.layout_var,
                                    values=["Grid", "Irregular", "Packed"], width=10)
        layout_combo.pack(side=tk.LEFT, padx=(5, 10))

        # Notes
        notes_frame = ttk.Frame(annotation_frame)
        notes_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(notes_frame, text="Notes:").pack(side=tk.LEFT)
        self.notes_var = tk.StringVar()
        notes_entry = ttk.Entry(
            notes_frame, textvariable=self.notes_var, width=50)
        notes_entry.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)

        # Save Button
        ttk.Button(annotation_frame, text="Save Annotation",
                   command=self.save_annotation).pack(pady=10)

        # Progress Info
        self.progress_var = tk.StringVar(value="No ground truth loaded")
        progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        progress_label.pack(pady=5)

    def load_existing_ground_truth(self):
        """Lädt existierende Ground Truth Daten"""
        if os.path.exists(self.annotation_file):
            try:
                with open(self.annotation_file, 'r') as f:
                    self.ground_truth_data = json.load(f)
                print(
                    f"Ground Truth geladen: {len(self.ground_truth_data)} Annotationen")
            except:
                self.ground_truth_data = {}
        else:
            self.ground_truth_data = {}

    def load_image(self):
        """Lädt ein einzelnes Spritesheet"""
        filename = filedialog.askopenfilename(
            title="Select Spritesheet",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if filename:
            self.display_image(filename)

    def load_random_sample(self):
        """Lädt zufälliges Sample aus input Ordner"""
        input_dir = "input"
        if not os.path.exists(input_dir):
            messagebox.showerror("Error", "Input directory not found!")
            return

        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
            image_files.extend(Path(input_dir).glob(ext))

        if not image_files:
            messagebox.showerror(
                "Error", "No images found in input directory!")
            return

        # Bevorzuge noch nicht annotierte Dateien
        unannotated = [f for f in image_files if str(
            f) not in self.ground_truth_data]
        if unannotated:
            selected_file = random.choice(unannotated)
        else:
            selected_file = random.choice(image_files)

        self.display_image(str(selected_file))

    def display_image(self, filename):
        """Zeigt Spritesheet in UI an"""
        try:
            self.current_filename = filename
            image = cv2.imread(filename)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Skaliere für Display
            h, w = image_rgb.shape[:2]
            max_size = 800
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image_rgb = cv2.resize(image_rgb, (new_w, new_h))

            # Convert to PhotoImage
            pil_image = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(pil_image)

            self.image_label.configure(image=photo)
            self.image_label.image = photo

            # Update UI mit existierenden Daten falls vorhanden
            if filename in self.ground_truth_data:
                data = self.ground_truth_data[filename]
                self.frame_count_var.set(str(data.get('frame_count', '')))
                self.complexity_var.set(data.get('complexity', 'Medium'))
                self.layout_var.set(data.get('layout', 'Grid'))
                self.notes_var.set(data.get('notes', ''))
            else:
                # Reset für neues Image
                self.frame_count_var.set('')
                self.complexity_var.set('Medium')
                self.layout_var.set('Grid')
                self.notes_var.set('')

            self.update_progress()

        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")

    def save_annotation(self):
        """Speichert aktuelle Annotation"""
        if not self.current_filename:
            messagebox.showerror("Error", "No image loaded!")
            return

        try:
            frame_count = int(self.frame_count_var.get())
            if frame_count <= 0:
                raise ValueError("Frame count must be positive")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid frame count!")
            return

        # Speichere Annotation
        self.ground_truth_data[self.current_filename] = {
            'frame_count': frame_count,
            'complexity': self.complexity_var.get(),
            'layout': self.layout_var.get(),
            'notes': self.notes_var.get(),
            'image_size': self.get_image_size(self.current_filename),
            'annotated_at': self.get_timestamp()
        }

        self.save_ground_truth()
        self.update_progress()
        messagebox.showinfo("Success", "Annotation saved!")

        # Automatisch nächstes zufälliges Sample laden
        self.load_random_sample()

    def get_image_size(self, filename):
        """Ermittelt Bildgröße"""
        try:
            image = cv2.imread(filename)
            h, w = image.shape[:2]
            return {'width': w, 'height': h}
        except:
            return {'width': 0, 'height': 0}

    def get_timestamp(self):
        """Aktuelle Zeit als String"""
        from datetime import datetime
        return datetime.now().isoformat()

    def save_ground_truth(self):
        """Speichert Ground Truth in JSON Datei"""
        try:
            with open(self.annotation_file, 'w') as f:
                json.dump(self.ground_truth_data, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Could not save ground truth: {e}")

    def update_progress(self):
        """Aktualisiert Progress-Info"""
        total = len(self.ground_truth_data)
        current_status = f"Annotated: {total} spritesheets"
        if self.current_filename and self.current_filename in self.ground_truth_data:
            current_status += " (Current: ANNOTATED)"
        elif self.current_filename:
            current_status += " (Current: NOT ANNOTATED)"
        self.progress_var.set(current_status)

    def export_report(self):
        """Exportiert Ground Truth Report"""
        if not self.ground_truth_data:
            messagebox.showerror("Error", "No ground truth data available!")
            return

        report = self.generate_report()

        filename = filedialog.asksaveasfilename(
            title="Save Ground Truth Report",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")]
        )

        if filename:
            try:
                if filename.endswith('.json'):
                    with open(filename, 'w') as f:
                        json.dump(report, f, indent=2)
                else:
                    with open(filename, 'w') as f:
                        f.write(self.format_report_text(report))
                messagebox.showinfo("Success", f"Report saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save report: {e}")

    def generate_report(self):
        """Generiert Ground Truth Report"""
        frame_counts = [data['frame_count']
                        for data in self.ground_truth_data.values()]
        complexities = [data['complexity']
                        for data in self.ground_truth_data.values()]
        layouts = [data['layout'] for data in self.ground_truth_data.values()]

        report = {
            'summary': {
                'total_annotated': len(self.ground_truth_data),
                'frame_count_stats': {
                    'min': min(frame_counts) if frame_counts else 0,
                    'max': max(frame_counts) if frame_counts else 0,
                    'mean': sum(frame_counts) / len(frame_counts) if frame_counts else 0,
                    'median': sorted(frame_counts)[len(frame_counts)//2] if frame_counts else 0
                },
                'complexity_distribution': {
                    'Simple': complexities.count('Simple'),
                    'Medium': complexities.count('Medium'),
                    'Complex': complexities.count('Complex')
                },
                'layout_distribution': {
                    'Grid': layouts.count('Grid'),
                    'Irregular': layouts.count('Irregular'),
                    'Packed': layouts.count('Packed')
                }
            },
            'detailed_annotations': self.ground_truth_data
        }

        return report

    def format_report_text(self, report):
        """Formatiert Report als Text"""
        summary = report['summary']
        text = f"""SPRITESHEET GROUND TRUTH REPORT
=====================================

SUMMARY:
--------
Total Annotated: {summary['total_annotated']}

Frame Count Statistics:
- Min: {summary['frame_count_stats']['min']}
- Max: {summary['frame_count_stats']['max']}
- Mean: {summary['frame_count_stats']['mean']:.2f}
- Median: {summary['frame_count_stats']['median']}

Background Complexity Distribution:
- Simple: {summary['complexity_distribution']['Simple']}
- Medium: {summary['complexity_distribution']['Medium']}
- Complex: {summary['complexity_distribution']['Complex']}

Layout Distribution:
- Grid: {summary['layout_distribution']['Grid']}
- Irregular: {summary['layout_distribution']['Irregular']}
- Packed: {summary['layout_distribution']['Packed']}

DETAILED ANNOTATIONS:
--------------------
"""
        for filename, data in report['detailed_annotations'].items():
            text += f"\nFile: {os.path.basename(filename)}\n"
            text += f"  Frames: {data['frame_count']}\n"
            text += f"  Complexity: {data['complexity']}\n"
            text += f"  Layout: {data['layout']}\n"
            if data['notes']:
                text += f"  Notes: {data['notes']}\n"

        return text

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = GroundTruthCreator()
    app.run()
