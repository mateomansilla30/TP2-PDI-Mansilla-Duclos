import cv2
import numpy as np
import matplotlib.pyplot as plt

class PCBComponentAnalyzer:
    """
    Analizador de componentes electrónicos en placas PCB mediante visión computacional
    Implementa algoritmos de segmentación y clasificación para detección automática
    """
    
    def __init__(self, image_path):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Error al cargar la imagen: {image_path}")
        
        self.image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Variables de procesamiento
        self.gray_image = None
        self.blurred_image = None
        self.edge_image = None
        self.processed_edges = None
        
        # Componentes detectados
        self.component_stats = None
        self.component_labels = None
        self.detected_components = []
        self.available_components = []
        self.resistor_list = []
        self.capacitor_list = []
        self.chip_list = []

    def _preprocess_image(self):
        """Preprocesamiento: conversión a escala de grises y filtrado Gaussiano"""
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        # Kernel 11x11 optimizado para reducción de ruido en PCBs
        self.blurred_image = cv2.GaussianBlur(self.gray_image, (11, 11), 1)
        return self.blurred_image
    
    def _detect_edges(self, low_thresh=50, high_thresh=80):
        """Detección de bordes mediante algoritmo Canny con umbrales optimizados"""
        if self.blurred_image is None:
            self._preprocess_image()
        
        self.edge_image = cv2.Canny(self.blurred_image, low_thresh, high_thresh)
        return self.edge_image
    
    def _enhance_edges(self):
        """Mejora de bordes usando operaciones morfológicas de cierre y dilatación"""
        if self.edge_image is None:
            self._detect_edges()
        
        # Cierre morfológico para conectar bordes fragmentados
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        closed_edges = cv2.morphologyEx(self.edge_image, cv2.MORPH_CLOSE, closing_kernel)
        
        # Dilatación controlada para engrosar líneas
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        self.processed_edges = cv2.dilate(closed_edges, dilation_kernel, iterations=1)
        
        return self.processed_edges
    
    def _find_connected_components(self, min_area_threshold=3200):
        """Análisis de componentes conectados con filtrado por área mínima"""
        if self.processed_edges is None:
            self._enhance_edges()
        
        # Análisis con conectividad 8 para máxima detección
        num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
            self.processed_edges, connectivity=8
        )
        
        self.component_labels = labels
        self.component_stats = stats
        
        # Filtrado de componentes válidos por área
        valid_components = []
        for i in range(1, num_components):
            x, y, w, h, area = stats[i]
            if area > min_area_threshold:
                component_data = {
                    'id': i,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': max(w, h) / min(w, h) if min(w, h) > 0 else 0,
                    'centroid': centroids[i],
                    'classified': False
                }
                valid_components.append(component_data)
        
        self.detected_components = valid_components
        self.available_components = valid_components.copy()
        return valid_components

    def _remove_classified_components(self, classified_components):
        """Actualización de listas tras clasificación de componentes"""
        classified_ids = [comp['id'] for comp in classified_components]
        self.available_components = [comp for comp in self.available_components 
                                   if comp['id'] not in classified_ids]
        
        for comp in self.detected_components:
            if comp['id'] in classified_ids:
                comp['classified'] = True

    def _classify_resistors(self):
        """
        Clasificación de resistencias basada en:
        - Análisis geométrico (aspect ratio >= 1.7)
        - Análisis colorimétrico (detección de marrón característico)
        """
        if not self.available_components:
            if not self.detected_components:
                self._find_connected_components()
        
        resistors = []
        
        for component in self.available_components:
            if component['classified']:
                continue
                
            x, y, w, h = component['bbox']
            area = component['area']
            aspect_ratio = component['aspect_ratio']
            
            # Criterio geométrico: forma alargada típica de resistencias
            if aspect_ratio >= 1.7 and area <= 10000:
                roi = self.image_rgb[y:y+h, x:x+w]
                
                # Análisis colorimétrico en espacio RGB
                target_color = np.array([200, 160, 100], dtype=np.uint8)
                tolerance = 50
                
                lower_bound = np.clip(target_color - tolerance, 0, 255)
                upper_bound = np.clip(target_color + tolerance, 0, 255)
                
                color_mask = cv2.inRange(roi, lower_bound, upper_bound)
                total_pixels = roi.size // 3
                matching_pixels = np.count_nonzero(color_mask)
                color_percentage = (matching_pixels / total_pixels) * 100
                
                # Umbral de coincidencia colorimétrica del 23%
                if color_percentage > 23:
                    component['color_match'] = color_percentage
                    resistors.append(component)
        
        self.resistor_list = resistors
        self._remove_classified_components(resistors)
        return resistors
    
    def _classify_capacitors(self):
        """
        Detección de capacitores mediante:
        - Transformada de Hough para detección circular
        - Análisis de luminancia interior
        """
        if not self.available_components:
            return []
        
        capacitors = []
        
        for component in self.available_components:
            if component['classified']:
                continue
                
            x, y, w, h = component['bbox']
            area = component['area']
            aspect_ratio = component['aspect_ratio']
            
            if area > 3200 and aspect_ratio < 1.9:
                roi = self.image_rgb[y:y+h, x:x+w]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                blurred_roi = cv2.GaussianBlur(gray_roi, (9, 9), 0)
                
                # Transformada de Hough con parámetros optimizados para capacitores
                circles = cv2.HoughCircles(
                    blurred_roi,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=250,
                    param1=70,      # Umbral Canny superior
                    param2=50,      # Umbral acumulador
                    minRadius=25,
                    maxRadius=200
                )
                
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    valid_circles = []
                    
                    for circle in circles:
                        cx, cy, radius = circle
                        
                        # Análisis de luminancia en región circular
                        mask = np.zeros(gray_roi.shape[:2], dtype=np.uint8)
                        cv2.circle(mask, (cx, cy), radius-5, 255, -1)
                        
                        circle_pixels = gray_roi[mask == 255]
                        avg_brightness = np.mean(circle_pixels)
                        
                        # Filtro por luminancia (capacitores típicamente claros)
                        if avg_brightness > 130:
                            # Clasificación por radio para categorización de tamaño
                            if radius < 100:
                                size_class = "PEQUEÑO"
                                color_code = (255, 0, 0)
                            elif radius < 150:
                                size_class = "MEDIANO"
                                color_code = (0, 255, 0)
                            else:
                                size_class = "GRANDE"
                                color_code = (0, 0, 255)
                            
                            circle_info = {
                                'center': (x + cx, y + cy),
                                'radius': radius,
                                'size_category': size_class,
                                'color': color_code,
                                'brightness': avg_brightness
                            }
                            valid_circles.append(circle_info)
                    
                    if valid_circles:
                        component['circles'] = valid_circles
                        capacitors.append(component)
        
        self.capacitor_list = capacitors
        self._remove_classified_components(capacitors)
        return capacitors
    
    def _classify_chips(self):
        """Detección de chips basada en criterio de área (>38000 píxeles)"""
        if not self.available_components:
            return []
        
        chips = []
        
        for component in self.available_components:
            if component['classified']:
                continue
                
            area = component['area']
            
            # Criterio de área para identificación de chips
            if area > 38000:
                chips.append(component)
        
        self.chip_list = chips
        self._remove_classified_components(chips)
        return chips

    def analyze_and_segment(self):
        """
        Pipeline completo de segmentación:
        1. Preprocesamiento y detección de bordes
        2. Análisis de componentes conectados
        3. Clasificación multiclase
        4. Visualización con anotaciones
        """
        # Pipeline de procesamiento de imagen
        self._preprocess_image()
        self._detect_edges()
        self._enhance_edges()
        self._find_connected_components()
        
        # Clasificación secuencial
        resistors = self._classify_resistors()
        capacitors = self._classify_capacitors()
        chips = self._classify_chips()
        
        # Generación de imagen anotada
        output_image = self.image_rgb.copy()
        
        # Anotación de resistencias
        for resistor in resistors:
            x, y, w, h = resistor['bbox']
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 255, 0), 6)
            cv2.putText(output_image, "RESISTOR", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Anotación de capacitores con codificación de color por tamaño
        for capacitor in capacitors:
            if 'circles' in capacitor:
                for circle in capacitor['circles']:
                    center = circle['center']
                    radius = circle['radius']
                    color = circle['color']
                    cv2.circle(output_image, center, radius, color, 6)
        
        # Anotación de chips
        for chip in chips:
            x, y, w, h = chip['bbox']
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 255), 6)
            cv2.putText(output_image, 'CHIP', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        # Visualización de resultados
        plt.figure(figsize=(15, 10))
        plt.imshow(output_image)
        title = f"SEGMENTACIÓN DE COMPONENTES\nResistores: {len(resistors)} | Capacitores: {len(capacitors)} | Chips: {len(chips)}"
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return output_image
    
    def classify_capacitors_by_size(self):
        """Clasificación granular de capacitores por análisis de radio"""
        if not self.capacitor_list:
            return None, None
        
        # Umbrales de clasificación por tamaño
        small_threshold = 80
        medium_threshold = 180
        
        size_distribution = {"PEQUEÑO": 0, "MEDIANO": 0, "GRANDE": 0}
        visualization = self.image_rgb.copy()
        
        for capacitor in self.capacitor_list:
            if 'circles' in capacitor:
                for circle in capacitor['circles']:
                    center = circle['center']
                    radius = circle['radius']
                    
                    # Reclasificación por umbrales actualizados
                    if radius < small_threshold:
                        size_category = "PEQUEÑO"
                        color = (255, 0, 0)
                    elif radius < medium_threshold:
                        size_category = "MEDIANO"
                        color = (0, 255, 0)
                    else:
                        size_category = "GRANDE"
                        color = (0, 0, 255)
                    
                    circle['size_category'] = size_category
                    circle['color'] = color
                    
                    size_distribution[size_category] += 1
                    cv2.circle(visualization, center, radius, color, 6)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(visualization)
        
        title = "CLASIFICACIÓN DE CAPACITORES POR TAMAÑO\n"
        title += f"Pequeños: {size_distribution['PEQUEÑO']} | "
        title += f"Medianos: {size_distribution['MEDIANO']} | "
        title += f"Grandes: {size_distribution['GRANDE']}"
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return visualization, size_distribution
    
    def count_resistors(self):
        """Conteo final de resistencias detectadas"""
        if not self.resistor_list:
            return 0
        
        resistor_count = len(self.resistor_list)
        print(f"CANTIDAD TOTAL DE RESISTENCIAS: {resistor_count}")
        return resistor_count
    
    def perform_complete_analysis(self):
        """Análisis completo con pipeline integrado"""
        segmentation_result = self.analyze_and_segment()
        classification_result = self.classify_capacitors_by_size()
        resistor_count = self.count_resistors()
        
        return {
            'segmentation': segmentation_result,
            'capacitor_classification': classification_result,
            'resistor_count': resistor_count
        }


if __name__ == "__main__":
    try:
        analyzer = PCBComponentAnalyzer("placa.png")
        results = analyzer.perform_complete_analysis()
        
    except Exception as e:
        print(f"Error durante el análisis: {e}")
        print("Verifique que el archivo 'placa.png' existe en el directorio actual")