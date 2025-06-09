import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob


def procesar_resistencia(imagen_path):
    """
    Procesa una imagen de resistencia y devuelve la vista superior del rectángulo azul.
    
    Args:
        imagen_path (str): Ruta de la imagen de entrada
    
    Returns:
        numpy.ndarray: Imagen transformada a vista superior, o None si hay error
    """
    img = cv2.imread(imagen_path)
    if img is None:
        print(f"Error: No se pudo cargar {imagen_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Suavizado gaussiano
    img_suavizada = cv2.GaussianBlur(img_rgb, (5, 5), 1.0)
    
    # Segmentación por color HSV para detectar azul
    img_hsv = cv2.cvtColor(img_suavizada, cv2.COLOR_RGB2HSV)
    azul_bajo = np.array([100, 50, 50])
    azul_alto = np.array([130, 255, 255])
    mascara_azul = cv2.inRange(img_hsv, azul_bajo, azul_alto)
    
    # Operaciones morfológicas para limpiar
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mascara_limpia = cv2.morphologyEx(mascara_azul, cv2.MORPH_OPEN, kernel)
    mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel)
    
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel_vertical)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contornos) == 0:
        print(f"Error: No se encontraron contornos en {imagen_path}")
        return None
    
    contorno_principal = max(contornos, key=cv2.contourArea)
    
    # Aproximación poligonal para obtener 4 esquinas
    epsilon = 0.02 * cv2.arcLength(contorno_principal, True)
    aproximacion = cv2.approxPolyDP(contorno_principal, epsilon, True)
    
    if len(aproximacion) != 4:
        rect = cv2.minAreaRect(contorno_principal)
        puntos_rect = cv2.boxPoints(rect)
        aproximacion = np.int32(puntos_rect).reshape(-1, 1, 2)
    
    # Ordenar los 4 puntos del rectángulo
    puntos = aproximacion.reshape(4, 2).astype(np.float32)
    puntos_ordenados = np.zeros((4, 2), dtype=np.float32)
    
    suma = puntos.sum(axis=1)
    diferencia = np.diff(puntos, axis=1)
    
    puntos_ordenados[0] = puntos[np.argmin(suma)]
    puntos_ordenados[2] = puntos[np.argmax(suma)]
    puntos_ordenados[1] = puntos[np.argmin(diferencia)]
    puntos_ordenados[3] = puntos[np.argmax(diferencia)]
    
    # Calcular dimensiones del rectángulo de salida
    ancho_a = np.sqrt(((puntos_ordenados[2][0] - puntos_ordenados[3][0]) ** 2) +
                      ((puntos_ordenados[2][1] - puntos_ordenados[3][1]) ** 2))
    ancho_b = np.sqrt(((puntos_ordenados[1][0] - puntos_ordenados[0][0]) ** 2) +
                      ((puntos_ordenados[1][1] - puntos_ordenados[0][1]) ** 2))
    ancho_max = max(int(ancho_a), int(ancho_b))
    
    alto_a = np.sqrt(((puntos_ordenados[1][0] - puntos_ordenados[2][0]) ** 2) +
                     ((puntos_ordenados[1][1] - puntos_ordenados[2][1]) ** 2))
    alto_b = np.sqrt(((puntos_ordenados[0][0] - puntos_ordenados[3][0]) ** 2) +
                     ((puntos_ordenados[0][1] - puntos_ordenados[3][1]) ** 2))
    alto_max = max(int(alto_a), int(alto_b))
    
    # Transformación de perspectiva
    puntos_destino = np.array([
        [0, 0],
        [ancho_max - 1, 0],
        [ancho_max - 1, alto_max - 1],
        [0, alto_max - 1]
    ], dtype=np.float32)
    
    matriz_transformacion = cv2.getPerspectiveTransform(puntos_ordenados, puntos_destino)
    img_transformada = cv2.warpPerspective(img_rgb, matriz_transformacion, (ancho_max, alto_max))
    
    return img_transformada

def procesar_imagen_individual(ruta_entrada, ruta_salida):
    """
    Procesa una sola imagen y la guarda.
    
    Args:
        ruta_entrada (str): Ruta de la imagen original
        ruta_salida (str): Ruta donde guardar el resultado
    
    Returns:
        bool: True si se procesó correctamente, False en caso contrario
    """
    resultado = procesar_resistencia(ruta_entrada)
    
    if resultado is not None:
        img_guardar = cv2.cvtColor(resultado, cv2.COLOR_RGB2BGR)
        cv2.imwrite(ruta_salida, img_guardar)
        print(f"Procesado: {os.path.basename(ruta_entrada)} -> {os.path.basename(ruta_salida)}")
        return True
    else:
        print(f"Error procesando: {os.path.basename(ruta_entrada)}")
        return False

def procesar_todas_las_resistencias(carpeta_entrada="Resistencias", carpeta_salida="Resistencias_output"):
    """
    Procesa todas las imágenes de resistencias en la carpeta especificada.
    
    Args:
        carpeta_entrada (str): Carpeta con las imágenes originales
        carpeta_salida (str): Carpeta donde guardar los resultados
    """
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    patrones = [
        os.path.join(carpeta_entrada, "R*_*.jpg"),
        os.path.join(carpeta_entrada, "R*_*.png"),
        os.path.join(carpeta_entrada, "R*_*.jpeg")
    ]
    
    archivos_encontrados = []
    for patron in patrones:
        archivos_encontrados.extend(glob.glob(patron))
    
    if not archivos_encontrados:
        print(f"No se encontraron imágenes en la carpeta '{carpeta_entrada}'")
        print("Formatos buscados: R*_*.jpg, R*_*.png, R*_*.jpeg")
        return
    
    print(f"Encontradas {len(archivos_encontrados)} imágenes para procesar")
    print(f"Carpeta de entrada: {carpeta_entrada}")
    print(f"Carpeta de salida: {carpeta_salida}")
    print("-" * 50)
    
    procesadas_exitosamente = 0
    
    for ruta_entrada in sorted(archivos_encontrados):
        nombre_archivo = os.path.basename(ruta_entrada)
        nombre_sin_ext = os.path.splitext(nombre_archivo)[0]
        nombre_salida = f"{nombre_sin_ext}_out.jpg"
        ruta_salida = os.path.join(carpeta_salida, nombre_salida)
        
        if procesar_imagen_individual(ruta_entrada, ruta_salida):
            procesadas_exitosamente += 1
    
    print("-" * 50)
    print(f"Proceso completado: {procesadas_exitosamente}/{len(archivos_encontrados)} imágenes procesadas exitosamente")

if __name__ == "__main__":
    procesar_todas_las_resistencias()

class AnalizadorResistencias:
    """
    Analizador de resistencias mediante visión por computadora.
    Detecta bandas de colores y calcula el valor de resistencia según estándar IEC 60062.
    
    Pipeline: Segmentación HSV → Detección Sobel → Análisis de picos → Clasificación → Cálculo
    """
    
    def __init__(self, ratio_distancia_tolerancia=0.15, umbral_intensidad=300):
        self.definiciones_colores = self._definir_colores()
        self.valores_colores = self._definir_valores_colores()
        self.ratio_distancia_tolerancia = ratio_distancia_tolerancia
        self.umbral_intensidad = umbral_intensidad
    
    def _definir_colores(self):
        """Rangos HSV calibrados para cada color estándar de resistencias."""
        return {
            'Negro': {
                'hsv_range': [(np.array([0, 0, 0]), np.array([180, 40, 30]))],
                'name': 'Negro'
            },
            'Marrón': {
                'hsv_range': [(np.array([5, 50, 20]), np.array([15, 200, 100]))],
                'name': 'Marrón'
            },
            'Rojo': {
                # Dos rangos por discontinuidad en 0°/360°
                'hsv_range': [
                    (np.array([0, 150, 100]), np.array([5, 255, 255])),
                    (np.array([175, 150, 100]), np.array([180, 255, 255]))
                ],
                'name': 'Rojo'
            },
            'Naranja': {
                'hsv_range': [(np.array([10, 120, 120]), np.array([20, 255, 255]))],
                'name': 'Naranja'
            },
            'Amarillo': {
                'hsv_range': [(np.array([20, 60, 60]), np.array([40, 255, 255]))],
                'name': 'Amarillo'
            },
            'Verde': {
                'hsv_range': [(np.array([36, 100, 80]), np.array([75, 255, 255]))],
                'name': 'Verde'
            },
            'Azul': {
                'hsv_range': [(np.array([90, 100, 80]), np.array([120, 255, 255]))],
                'name': 'Azul'
            },
            'Violeta': {
                'hsv_range': [(np.array([125, 100, 80]), np.array([155, 255, 255]))],
                'name': 'Violeta'
            },
            'Gris': {
                'hsv_range': [(np.array([0, 0, 50]), np.array([180, 30, 160]))],
                'name': 'Gris'
            },
            'Blanco': {
                'hsv_range': [(np.array([0, 0, 200]), np.array([180, 20, 255]))],
                'name': 'Blanco'
            },
            'Plata': {
                'hsv_range': [(np.array([0, 0, 120]), np.array([180, 15, 200]))],
                'name': 'Plata'
            }
        }
    
    def _definir_valores_colores(self):
        """Mapeo según estándar IEC 60062 para códigos de colores."""
        return {
            'Negro': {'valor': 0, 'multiplicador': 1, 'tolerancia': None},
            'Marrón': {'valor': 1, 'multiplicador': 10, 'tolerancia': 1},
            'Rojo': {'valor': 2, 'multiplicador': 100, 'tolerancia': 2},
            'Naranja': {'valor': 3, 'multiplicador': 1000, 'tolerancia': None},
            'Amarillo': {'valor': 4, 'multiplicador': 10000, 'tolerancia': None},
            'Verde': {'valor': 5, 'multiplicador': 100000, 'tolerancia': 0.5},
            'Azul': {'valor': 6, 'multiplicador': 1000000, 'tolerancia': 0.25},
            'Violeta': {'valor': 7, 'multiplicador': 10000000, 'tolerancia': 0.1},
            'Gris': {'valor': 8, 'multiplicador': 100000000, 'tolerancia': 0.05},
            'Blanco': {'valor': 9, 'multiplicador': 1000000000, 'tolerancia': None},
            'Dorado': {'valor': None, 'multiplicador': 0.1, 'tolerancia': 5},
            'Plata': {'valor': None, 'multiplicador': 0.01, 'tolerancia': 10}
        }
    
    def calcular_valor_resistencia(self, colores_valor, colores_tolerancia=None):
        """Fórmula estándar: (D1*10 + D2) * M donde D1, D2 son dígitos y M multiplicador."""
        if len(colores_valor) < 3:
            return {
                'valor_ohms': None,
                'valor_formateado': 'Insuficientes bandas',
                'tolerancia': None,
                'error': 'Se necesitan al menos 3 bandas de valor'
            }
        
        try:
            primer_digito = self.valores_colores[colores_valor[0]]['valor']
            segundo_digito = self.valores_colores[colores_valor[1]]['valor']
            multiplicador = self.valores_colores[colores_valor[2]]['multiplicador']
            
            if primer_digito is None or segundo_digito is None:
                raise ValueError("Colores de dígitos inválidos")
            
            valor_base = (primer_digito * 10 + segundo_digito) * multiplicador
            
            tolerancia = None
            if colores_tolerancia and len(colores_tolerancia) > 0:
                color_tolerancia = colores_tolerancia[0]
                if color_tolerancia in self.valores_colores:
                    tolerancia = self.valores_colores[color_tolerancia]['tolerancia']
            
            valor_formateado = self._formatear_valor(valor_base)
            
            return {
                'valor_ohms': valor_base,
                'valor_formateado': valor_formateado,
                'tolerancia': tolerancia,
                'error': None
            }
            
        except (KeyError, ValueError) as e:
            return {
                'valor_ohms': None,
                'valor_formateado': 'Error en cálculo',
                'tolerancia': None,
                'error': str(e)
            }
    
    def _formatear_valor(self, valor_ohms):
        """Formateo con unidades ingenieriles (Ω, kΩ, MΩ)."""
        if valor_ohms >= 1_000_000:
            return f"{valor_ohms / 1_000_000:.1f}MΩ"
        elif valor_ohms >= 1_000:
            return f"{valor_ohms / 1_000:.1f}kΩ"
        else:
            return f"{valor_ohms:.1f}Ω"
    
    def _eliminar_fondo(self, img_rgb):
        """
        Segmentación de fondo azul mediante umbralización HSV y morfología.
        Retorna imagen limpia y máscara de resistencia.
        """
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        # Umbralización para fondo azul específico
        mascara_azul = cv2.inRange(img_hsv, np.array([100, 50, 50]), np.array([125, 255, 255]))
        mascara_resistencia = cv2.bitwise_not(mascara_azul)
        
        # Operaciones morfológicas: cerrado + apertura para limpieza
        kernel = np.ones((31, 3), np.uint8)
        mascara_resistencia = cv2.morphologyEx(mascara_resistencia, cv2.MORPH_CLOSE, kernel)
        mascara_resistencia = cv2.morphologyEx(mascara_resistencia, cv2.MORPH_OPEN, kernel)
        
        # Extracción del componente conexo principal
        contornos, _ = cv2.findContours(mascara_resistencia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contornos:
            contorno_principal = max(contornos, key=cv2.contourArea)
            mascara_limpia = np.zeros_like(mascara_resistencia)
            cv2.fillPoly(mascara_limpia, [contorno_principal], 255)
            
            img_limpia = img_rgb.copy()
            img_limpia[mascara_limpia == 0] = [255, 255, 255]
            
            return img_limpia, mascara_limpia
        
        return img_rgb, np.ones(img_rgb.shape[:2], dtype=np.uint8) * 255
    
    def _obtener_region_bandas(self, mascara, margen_h=0.14, margen_v=0.15):
        """ROI excluyendo terminales metálicas mediante márgenes proporcionales."""
        coordenadas = np.column_stack(np.where(mascara > 0))
        if len(coordenadas) == 0:
            return None
        
        y_min, x_min = coordenadas.min(axis=0)
        y_max, x_max = coordenadas.max(axis=0)
        
        ancho, alto = x_max - x_min, y_max - y_min
        margen_x, margen_y = int(ancho * margen_h), int(alto * margen_v)
        
        return {
            'x_inicio': x_min + margen_x,
            'x_fin': x_max - margen_x,
            'y_inicio': y_min + margen_y,
            'y_fin': y_max - margen_y
        }
    
    def _encontrar_bandas(self, img_rgb, mascara):
        """
        Detección mediante filtro Sobel-X y análisis de picos en proyección vertical.
        Implementa supresión no-máxima para eliminar detecciones duplicadas.
        """
        region = self._obtener_region_bandas(mascara)
        if not region:
            return []
        
        img_gris = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_gris[mascara == 0] = 255
        
        region_bandas = img_gris[region['y_inicio']:region['y_fin'], 
                                region['x_inicio']:region['x_fin']]
        
        # Filtro Sobel horizontal para bordes verticales
        sobel_x = cv2.Sobel(region_bandas, cv2.CV_64F, 1, 0, ksize=1)
        sobel_x = np.uint8(np.absolute(sobel_x))
        
        # Proyección vertical y suavizado
        perfil = np.sum(sobel_x, axis=0)
        kernel = np.ones(40) / 40
        perfil_suave = np.convolve(perfil, kernel, mode='same')
        
        # Detección de máximos locales
        candidatos = []
        for i in range(1, len(perfil_suave) - 1):
            if (perfil_suave[i] > perfil_suave[i-1] and 
                perfil_suave[i] > perfil_suave[i+1] and 
                perfil_suave[i] > self.umbral_intensidad):
                pos_abs = region['x_inicio'] + i
                candidatos.append((pos_abs, perfil_suave[i]))
        
        # Supresión no-máxima con distancia mínima empírica
        ancho = region['x_fin'] - region['x_inicio']
        distancia_min = ancho // 5.25
        picos_filtrados = []
        
        for pico in sorted(candidatos, key=lambda x: x[1], reverse=True):
            if not any(abs(pico[0] - existente[0]) < distancia_min for existente in picos_filtrados):
                picos_filtrados.append(pico)
                
        return sorted([pico[0] for pico in picos_filtrados])[:4]

    def _analizar_distancias_bandas(self, posiciones_bandas):
        """
        Separación valor/tolerancia mediante análisis estadístico de gaps.
        Detecta separación atípica que indica banda de tolerancia aislada.
        """
        if len(posiciones_bandas) < 3:
            return posiciones_bandas, []
        
        distancias = []
        for i in range(len(posiciones_bandas) - 1):
            distancias.append(posiciones_bandas[i+1] - posiciones_bandas[i])
        
        if len(distancias) >= 2:
            distancia_promedio = np.mean(distancias[:-1])
            
            # Gap significativo al final
            if len(distancias) >= 3 and distancias[-1] > distancia_promedio * (1 + self.ratio_distancia_tolerancia):
                return posiciones_bandas[:-1], [posiciones_bandas[-1]]
            
            # Gap al inicio
            elif distancias[0] > distancia_promedio * (1 + self.ratio_distancia_tolerancia):
                return posiciones_bandas[1:], [posiciones_bandas[0]]
        
        return posiciones_bandas, []
    
    def _detectar_color(self, region_hsv, nombre_color, config_color):
        """Conteo de píxeles coincidentes para múltiples rangos HSV."""
        pixeles_totales = 0
        
        for hsv_bajo, hsv_alto in config_color['hsv_range']:
            mascara = cv2.inRange(region_hsv, hsv_bajo, hsv_alto)
            pixeles_totales += cv2.countNonZero(mascara)
        
        return pixeles_totales
    
    def _identificar_colores_bandas(self, img_rgb, mascara, posiciones_bandas):
        """
        Clasificación por máxima verosimilitud en ROI centrada en cada banda.
        Ventana de análisis: 5 píxeles horizontales por altura de resistencia.
        """
        if not posiciones_bandas:
            return []
        
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        colores = []
        ancho_banda = 5
        
        for pos_x in posiciones_bandas:
            x_inicio = max(0, pos_x - ancho_banda // 2)
            x_fin = min(img_rgb.shape[1], pos_x + ancho_banda // 2)
            
            coords_y = np.where(mascara[:, pos_x] > 0)[0]
            if len(coords_y) > 0:
                y_inicio, y_fin = coords_y[0], coords_y[-1]
            else:
                y_inicio = img_rgb.shape[0] // 3
                y_fin = img_rgb.shape[0] * 2 // 3
            
            region_hsv = img_hsv[y_inicio:y_fin, x_inicio:x_fin]
            
            mejor_coincidencia = 'Desconocido'
            max_pixeles = 0
            
            for nombre_color, config_color in self.definiciones_colores.items():
                pixeles = self._detectar_color(region_hsv, nombre_color, config_color)
                if pixeles > max_pixeles:
                    max_pixeles = pixeles
                    mejor_coincidencia = config_color['name']
            
            colores.append(mejor_coincidencia)
        
        return colores
    
    def _asignar_color_tolerancia(self, posiciones_tolerancia, todos_colores):
        """Heurística: asigna dorado (5%) como tolerancia estándar."""
        if not posiciones_tolerancia:
            return []
        
        return ['Dorado'] * len(posiciones_tolerancia)
    
    def _visualizar_resultado(self, img_original, img_limpia, posiciones_valor, posiciones_tolerancia, 
                            colores_valor, colores_tolerancia, nombre_archivo, info_calculo):
        """Visualización diagnóstica con overlay de detecciones y resultados."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.imshow(img_original)
        ax1.set_title(f'Original: {nombre_archivo}')
        ax1.axis('off')
        
        img_resultado = img_limpia.copy()
        
        # Overlay bandas de valor (rojo)
        for i, pos in enumerate(posiciones_valor):
            cv2.line(img_resultado, (pos, 0), (pos, img_resultado.shape[0]), (0, 0, 255), 3)
            if i < len(colores_valor):
                etiqueta = f"V{i+1}: {colores_valor[i]}"
                cv2.putText(img_resultado, etiqueta, (pos-40, 30+i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Overlay bandas de tolerancia (verde)
        for i, pos in enumerate(posiciones_tolerancia):
            cv2.line(img_resultado, (pos, 0), (pos, img_resultado.shape[0]), (0, 255, 0), 3)
            if i < len(colores_tolerancia):
                etiqueta = f"T: {colores_tolerancia[i]}"
                cv2.putText(img_resultado, etiqueta, (pos-40, 120+i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        ax2.imshow(img_resultado)
        
        if len(colores_valor) >= 3:
            valor_texto = f"Valor: {info_calculo['valor_formateado']}"
            if info_calculo['tolerancia']:
                valor_texto += f" ±{info_calculo['tolerancia']}%"
            status = "✓"
        else:
            valor_texto = f"Solo {len(colores_valor)} bandas detectadas"
            status = "⚠"
        
        ax2.set_title(f'{status} {nombre_archivo}\n{valor_texto}')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analizar_resistencia_individual(self, ruta_imagen):
        """Pipeline completo: carga → segmentación → detección → clasificación → cálculo."""
        img = cv2.imread(ruta_imagen)
        if img is None:
            print(f"Error: No se pudo cargar {ruta_imagen}")
            return None
        
        nombre_archivo = os.path.basename(ruta_imagen)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_limpia, mascara = self._eliminar_fondo(img_rgb)
        todas_posiciones_bandas = self._encontrar_bandas(img_limpia, mascara)
        
        posiciones_valor, posiciones_tolerancia = self._analizar_distancias_bandas(todas_posiciones_bandas)
        
        colores_valor = self._identificar_colores_bandas(img_limpia, mascara, posiciones_valor)
        
        if posiciones_tolerancia:
            colores_tolerancia = self._asignar_color_tolerancia(posiciones_tolerancia, colores_valor)
        else:
            colores_tolerancia = []
        
        info_calculo = self.calcular_valor_resistencia(colores_valor, colores_tolerancia)
        
        self._visualizar_resultado(img_rgb, img_limpia, posiciones_valor, posiciones_tolerancia, 
                                 colores_valor, colores_tolerancia, nombre_archivo, info_calculo)
        
        return {
            'nombre_archivo': nombre_archivo,
            'bandas_totales': len(todas_posiciones_bandas),
            'bandas_valor': len(posiciones_valor),
            'bandas_tolerancia': len(posiciones_tolerancia),
            'colores_valor': colores_valor[:3] if len(colores_valor) >= 3 else colores_valor,
            'colores_tolerancia': colores_tolerancia,
            'posiciones_valor': posiciones_valor,
            'posiciones_tolerancia': posiciones_tolerancia,
            'calculo': info_calculo
        }
    
    def analizar_todas_resistencias(self, ruta_carpeta="Resistencias_output"):
        """Procesamiento batch con estadísticas de rendimiento."""
        patron = os.path.join(ruta_carpeta, "R*_a_out.jpg")
        archivos_imagen = glob.glob(patron)
        
        if not archivos_imagen:
            print(f"No se encontraron imágenes con formato Rx_a_out.jpg en {ruta_carpeta}")
            return []
        
        resultados = []
        print(f"Analizando {len(archivos_imagen)} resistencias...")
        print(f"Umbral de intensidad: {self.umbral_intensidad}")
        print(f"Ratio de distancia para tolerancia: {self.ratio_distancia_tolerancia}")
        print("=" * 80)
        
        for ruta_imagen in sorted(archivos_imagen):
            resultado = self.analizar_resistencia_individual(ruta_imagen)
            if resultado:
                resultados.append(resultado)
                
                nombre_archivo = resultado['nombre_archivo']
                colores_valor = resultado['colores_valor']
                info_calculo = resultado['calculo']
                
                status = "✓" if len(colores_valor) >= 3 else "⚠"
                
                if info_calculo['valor_formateado'] != 'Error en cálculo':
                    valor_texto = info_calculo['valor_formateado']
                    if info_calculo['tolerancia']:
                        valor_texto += f" ±{info_calculo['tolerancia']}%"
                else:
                    valor_texto = "Error en cálculo"
                
                print(f"{status} {nombre_archivo}: {valor_texto}")
        
        print("=" * 80)
        print(f"Análisis completado: {len(resultados)} resistencias procesadas")
             
        return resultados

if __name__ == "__main__":
    print("=== PROCESAMIENTO COMPLETO DE RESISTENCIAS ===")
    print("Paso 1: Preprocesando imágenes originales...")
    
    procesar_todas_las_resistencias()
    
    print("\n" + "="*50)
    print("Paso 2: Analizando resistencias procesadas...")
    
    analizador = AnalizadorResistencias(
        ratio_distancia_tolerancia=0.15,
        umbral_intensidad=200
    )
    
    resultados = analizador.analizar_todas_resistencias()
    
    print(f"\n=== PROCESO COMPLETO TERMINADO ===")
    print(f"Total de resistencias analizadas: {len(resultados)}")