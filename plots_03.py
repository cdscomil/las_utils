import laspy
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

class AdvancedLASViewer:
    def __init__(self, las_file_path):
        self.las_file_path = las_file_path
        self.pcd = None
        self.original_pcd = None
        self.load_las_file()
    
    def load_las_file(self):
        """Загрузка LAS-файла"""
        print(f"Загрузка: {self.las_file_path}")
        las = laspy.read(self.las_file_path)
        
        points = np.vstack((las.x, las.y, las.z)).transpose()
        
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        
        # Раскраска по высоте
        z_norm = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())
        colors = self.height_colormap(z_norm)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Сохраняем оригинал
        self.original_pcd = o3d.geometry.PointCloud(self.pcd)
        
        print(f"Точек загружено: {len(points)}")
    
    def height_colormap(self, values):
        """Цветовая карта по высоте"""
        colors = np.zeros((len(values), 3))
        colors[:, 0] = values
        colors[:, 1] = 1 - np.abs(values - 0.5) * 2
        colors[:, 2] = 1 - values
        return colors
    
    def downsample(self, voxel_size=0.5):
        """Уменьшение плотности точек"""
        print(f"Прореживание с размером вокселя: {voxel_size}")
        self.pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Точек после прореживания: {len(self.pcd.points)}")
    
    def remove_outliers(self, nb_neighbors=20, std_ratio=2.0):
        """Удаление выбросов"""
        print("Удаление выбросов...")
        self.pcd, ind = self.pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        print(f"Точек после фильтрации: {len(self.pcd.points)}")
    
    def crop_by_bounds(self, x_range=None, y_range=None, z_range=None):
        """Обрезка по границам"""
        points = np.asarray(self.pcd.points)
        mask = np.ones(len(points), dtype=bool)
        
        if x_range:
            mask &= (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
        if y_range:
            mask &= (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
        if z_range:
            mask &= (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
        
        self.pcd = self.pcd.select_by_index(np.where(mask)[0])
        print(f"Точек после обрезки: {len(self.pcd.points)}")
    
    def reset(self):
        """Восстановление оригинального облака"""
        self.pcd = o3d.geometry.PointCloud(self.original_pcd)
        print("Облако восстановлено")
    
    def visualize_interactive(self):
        """Интерактивная визуализация"""
        # SketchUp-like навигация:
        # - Зажатый скролл (MMB) + движение: вращение (orbit)
        # - Зажатая ЛКМ + движение: перемещение (pan)
        # - Колесо: zoom
        print("\n=== ИНТЕРАКТИВНЫЙ ПРОСМОТР (SketchUp) ===")
        print("Управление мышью:")
        print("  MMB(скролл) + движение: Вращение")
        print("  ЛКМ + движение: Перемещение")
        print("  Колесо: Масштаб")
        print("Закрыть окно: Alt+F4 / крестик")
        print("=========================================\n")

        # GUI-рендерер даёт полноценные callbacks мыши. Если вдруг GUI недоступен,
        # делаем fallback на стандартное окно (там ремап кнопок невозможен).
        try:
            from open3d.visualization import gui, rendering
        except Exception:
            print("GUI Open3D недоступен — использую стандартное управление Open3D.")
            o3d.visualization.draw_geometries(
                [self.pcd],
                window_name="Advanced LAS Viewer (fallback)",
                width=1280,
                height=720,
                point_show_normal=False,
            )
            return

        def _normalize(v: np.ndarray) -> np.ndarray:
            n = float(np.linalg.norm(v))
            if n == 0.0:
                return v
            return v / n

        def _rotate_rodrigues(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
            # Rodrigues rotation formula
            axis = _normalize(axis)
            c = float(np.cos(angle_rad))
            s = float(np.sin(angle_rad))
            return v * c + np.cross(axis, v) * s + axis * (np.dot(axis, v)) * (1.0 - c)

        # Камера: храним состояние сами, чтобы не зависеть от внутренних контролов SceneWidget.
        bbox = self.pcd.get_axis_aligned_bounding_box()
        center = np.asarray(bbox.get_center(), dtype=float)
        extent = np.asarray(bbox.get_extent(), dtype=float)
        radius = float(np.linalg.norm(extent) * 0.5)
        if not np.isfinite(radius) or radius <= 0:
            radius = 1.0

        world_up = np.array([0.0, 0.0, 1.0], dtype=float)  # LAS: Z вверх
        eye = center + np.array([0.0, -2.5 * radius, 1.2 * radius], dtype=float)

        app = gui.Application.instance
        app.initialize()

        win = app.create_window("Advanced LAS Viewer", 1280, 720)
        widget3d = gui.SceneWidget()
        widget3d.scene = rendering.Open3DScene(win.renderer)
        # Светлый фон (как в SketchUp) — контрастнее для точек
        widget3d.scene.set_background([0.95, 0.95, 0.95, 1.0])
        widget3d.scene.show_axes(True)

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        # Для больших облаков LAS 1-2px часто выглядит как "точка"
        mat.point_size = 4.0
        widget3d.scene.add_geometry("pcd", self.pcd, mat)

        # Стартовая камера: пусть Open3D сам "обрамит" bbox (правильные near/far/FOV)
        widget3d.setup_camera(60.0, bbox, center.astype(np.float32))
        # И синхронизируем наше состояние с этой камерой, чтобы дальнейшие orbit/pan были стабильными
        try:
            # view matrix (world->cam), invert to get camera position in world
            V = np.asarray(widget3d.scene.camera.get_view_matrix(), dtype=float)
            V_inv = np.linalg.inv(V)
            eye = V_inv[:3, 3].copy()
        except Exception:
            # fallback: оставляем ранее вычисленный eye
            widget3d.scene.camera.look_at(center, eye, world_up)

        # Состояние мыши
        state = {
            "last_x": None,
            "last_y": None,
        }

        def _apply_camera():
            widget3d.scene.camera.look_at(center, eye, world_up)
            widget3d.force_redraw()

        def _on_mouse(event: gui.MouseEvent) -> bool:
            nonlocal center, eye

            if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
                state["last_x"] = event.x
                state["last_y"] = event.y
                return True

            if event.type == gui.MouseEvent.Type.BUTTON_UP:
                state["last_x"] = event.x
                state["last_y"] = event.y
                return True

            if event.type == gui.MouseEvent.Type.DRAG:
                if state["last_x"] is None or state["last_y"] is None:
                    state["last_x"] = event.x
                    state["last_y"] = event.y
                    return True

                dx = float(event.x - state["last_x"])
                dy = float(event.y - state["last_y"])
                state["last_x"] = event.x
                state["last_y"] = event.y

                v = eye - center
                dist = float(np.linalg.norm(v))
                if not np.isfinite(dist) or dist <= 1e-9:
                    dist = 1.0

                # MMB: orbit/rotate
                if event.is_button_down(gui.MouseButton.MIDDLE):
                    yaw = -dx * 0.005
                    pitch = -dy * 0.005

                    v = _rotate_rodrigues(v, world_up, yaw)
                    right = np.cross(world_up, v)
                    if float(np.linalg.norm(right)) < 1e-9:
                        right = np.array([1.0, 0.0, 0.0], dtype=float)
                    v = _rotate_rodrigues(v, right, pitch)
                    eye = center + v
                    _apply_camera()
                    return True

                # LMB: pan
                if event.is_button_down(gui.MouseButton.LEFT):
                    forward = _normalize(center - eye)
                    right = _normalize(np.cross(forward, world_up))
                    up_cam = _normalize(np.cross(right, forward))

                    # Скорость панорамирования привязана к дистанции до сцены
                    pan_scale = dist * 0.0015
                    t = (-dx) * pan_scale * right + (dy) * pan_scale * up_cam

                    center = center + t
                    eye = eye + t
                    _apply_camera()
                    return True

                return False

            if event.type == gui.MouseEvent.Type.WHEEL:
                # Zoom-to-cursor: приближаем/отдаляем вдоль луча под мышью.
                # Без depth-buffer'а берём опорную плоскость на "глубине" текущего center
                # (плоскость через center, нормаль = направление взгляда).
                w = int(max(1, widget3d.frame.width))
                h = int(max(1, widget3d.frame.height))

                P = np.asarray(widget3d.scene.camera.get_projection_matrix(), dtype=float)
                V = np.asarray(widget3d.scene.camera.get_view_matrix(), dtype=float)
                PV = P @ V
                try:
                    PV_inv = np.linalg.inv(PV)
                except Exception:
                    return False

                # NDC: x,y in [-1, 1], OpenGL z in [-1, 1]
                ndc_x = (2.0 * ((float(event.x) + 0.5) / w)) - 1.0
                ndc_y = 1.0 - (2.0 * ((float(event.y) + 0.5) / h))

                near_clip = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=float)
                far_clip = np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=float)

                near_world = PV_inv @ near_clip
                far_world = PV_inv @ far_clip
                if abs(float(near_world[3])) < 1e-12 or abs(float(far_world[3])) < 1e-12:
                    return False
                near_world = near_world[:3] / float(near_world[3])
                far_world = far_world[:3] / float(far_world[3])

                # Луч из глаза через курсор
                ray_dir = _normalize(far_world - eye)
                view_dir = _normalize(center - eye)
                denom = float(np.dot(ray_dir, view_dir))
                if abs(denom) < 1e-8:
                    pivot = center
                else:
                    t = float(np.dot(center - eye, view_dir) / denom)
                    pivot = center if t <= 0.0 else (eye + ray_dir * t)

                # Нормализуем wheel шаг (Windows часто даёт 120 на щелчок)
                dy = float(event.wheel_dy)
                if (not event.wheel_is_trackpad) and abs(dy) >= 20.0:
                    dy /= 120.0

                # Мягкий zoom: dy>0 обычно "вверх" -> zoom in
                k = 0.18 if event.wheel_is_trackpad else 0.30
                zoom = float(np.exp(-dy * k))

                dist = float(np.linalg.norm(pivot - eye))
                if not np.isfinite(dist) or dist <= 1e-9:
                    dist = 1.0

                new_dist = dist * zoom
                new_dist = float(np.clip(new_dist, radius * 0.02, radius * 200.0))

                # Dolly вдоль луча к pivot, сохраняя pivot под курсором:
                # двигаем и eye, и center на одинаковый вектор.
                delta = dist - new_dist  # >0 приближаем
                move = _normalize(pivot - eye) * delta
                eye = eye + move
                center = center + move

                _apply_camera()
                return True

            return False

        widget3d.set_on_mouse(_on_mouse)

        win.add_child(widget3d)

        def _on_layout(layout_context):
            widget3d.frame = win.content_rect

        win.set_on_layout(_on_layout)
        app.run()
    
    def save_processed(self, output_path):
        """Сохранение обработанного облака"""
        o3d.io.write_point_cloud(output_path, self.pcd)
        print(f"Сохранено в: {output_path}")

# Использование
if __name__ == "__main__":

#    las_file = "PO_Goodline_AFS_05.08.2025.las"
    las_file = "out.las"
    
    viewer = AdvancedLASViewer(las_file)
    
    # Опциональная обработка
    # viewer.downsample(voxel_size=0.5)
    # viewer.remove_outliers()
    # viewer.crop_by_bounds(z_range=[0, 100])
    
    # Визуализация
    viewer.visualize_interactive()
    
    # Сохранение
    # viewer.save_processed("processed.ply")
