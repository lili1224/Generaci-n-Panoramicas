import cv2
import numpy as np
import glob

def resize_images_to_height(images, reference_height):
    resized_images = []
    for image in images:
        height, width = image.shape[:2]
        scale_ratio = reference_height / height
        new_width = int(width * scale_ratio)
        resized_image = cv2.resize(image, (new_width, reference_height))
        resized_images.append(resized_image)
    return resized_images

def crop_panorama(panorama):
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cropped_panorama = panorama[y:y+h, x:x+w]
        return cropped_panorama
    return panorama

# Cargar las imágenes desde una carpeta
image_paths = sorted(glob.glob('images/*.jpg'))
images = [cv2.imread(image_path) for image_path in image_paths]

# Asegurarse de que las imágenes han sido cargadas correctamente
if not images:
    print("No se encontraron imágenes en la carpeta especificada.")
    exit()

# Redimensionar todas las imágenes para que tengan la misma altura que la primera imagen
reference_height = images[0].shape[0]
images = resize_images_to_height(images, reference_height)

# Inicializar el objeto de stitcher de OpenCV
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

# Realizar el empalme (stitching)
status, panorama = stitcher.stitch(images)

# Verificar el estado del empalme y actuar en consecuencia
if status == cv2.Stitcher_OK:
    # Recortar la panorámica para eliminar los escalones
    cropped_panorama = crop_panorama(panorama)

    # Mostrar la panorámica resultante
    cv2.imshow('Panorámica', cropped_panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar la panorámica en un archivo
    cv2.imwrite('panoramica_resultante_cropped.jpg', cropped_panorama)
else:
    print(f'El empalme falló con el código de estado: {status}')
    if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        print("Error: Se necesitan más imágenes para crear la panorámica.")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        print("Error: Fallo en la estimación de la homografía.")
    elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        print("Error: Fallo en el ajuste de los parámetros de la cámara.")
    else:
        print("Error: Código de error desconocido.")
