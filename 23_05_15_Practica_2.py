import tkinter as tk

# Ordenamiento de burbuja (Bubble Sort)
def bubbleSort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def ejecutar_ordenamiento_bubble():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = bubbleSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Bubble Sort: \n" + str(lista_ordenada))

# Ordenamiento por selección (Selection Sort)
def selectionSort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
        
def ejecutar_ordenamiento_sort():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = selectionSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Selection Sort: \n" + str(lista_ordenada))

# Ordenamiento por inserción (Insertion Sort)
def insertion1Sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and key < arr[j] :
                arr[j+1] = arr[j]
                j -= 1
        arr[j+1] = key
    return arr

def ejecutar_ordenamiento_insertion():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = insertion1Sort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Insertion Sort: \n" + str(lista_ordenada))
    
# Ordenamiento por mezcla (Merge Sort)
def mergeSort(arr):
    if len(arr) > 1:
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]
        mergeSort(L)
        mergeSort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
    return arr

def ejecutar_ordenamiento_mergeSort():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(float, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = mergeSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Merge Sort: \n" + str(lista_ordenada))
    
# Ordenamiento rápido (Quick Sort)

def quickSort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        lesser = [x for x in arr[1:] if x <= pivot]
        greater = [x for x in arr[1:] if x > pivot]
        return quickSort(lesser) + [pivot] + quickSort(greater)
    
def ejecutar_ordenamiento_quick():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = quickSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Quick Sort: \n" + str(lista_ordenada))

# Ordenamiento por montículos (Heap Sort)

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[i] < arr[left]:
        largest = left

    if right < n and arr[largest] < arr[right]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heapSort(arr):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

def ejecutar_ordenamiento_heap():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = heapSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Heap Sort: \n" + str(lista_ordenada))

# Ordenamiento por cuenta (Counting Sort)

def counting1Sort(arr):
    # Encontrar el valor máximo en la lista
    max_value = max(arr)

    # Crear una lista de conteo con tamaño max_value + 1, inicializada en cero
    count = [0] * (max_value + 1)

    # Contar la frecuencia de cada elemento en arr
    for num in arr:
        count[num] += 1

    # Reconstruir la lista ordenada utilizando los valores y frecuencias del conteo
    sorted_arr = []
    for i in range(len(count)):
        sorted_arr.extend([i] * count[i])

    return sorted_arr

def ejecutar_ordenamiento_counting():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = counting1Sort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Counting Sort: \n" + str(lista_ordenada))

# Ordenamiento de cubetas (Bucket Sort)

def bucketSort(arr):
    # Crear un arreglo de cubetas vacías
    num_buckets = len(arr)
    buckets = [[] for _ in range(num_buckets)]

    # Colocar los elementos en las cubetas correspondientes
    for num in arr:
        index = int(num * num_buckets)
        buckets[index].append(num)

    # Ordenar cada cubeta individualmente
    for bucket in buckets:
        bucket.sort()

    # Concatenar las cubetas ordenadas en una lista
    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(bucket)

    return sorted_arr

def ejecutar_ordenamiento_bucket():
    # Obtener los datos ingresados por el usuario
    datos = entrada2.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(float, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = bucketSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado2.config(text="Datos ordenados por metodo Bucket Sort: \n" + str(lista_ordenada))

# Ordenamiento radix (Radix Sort)

def countingSort(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    for i in range(n):
        arr[i] = output[i]

def radixSort(arr):
    max_value = max(arr)
    exp = 1

    while max_value // exp > 0:
        countingSort(arr, exp)
        exp *= 10
    return arr

def ejecutar_ordenamiento_radix():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = radixSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Radix Sort: \n" + str(lista_ordenada))

# Ordenamiento por árbol (Tree Sort)

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def insertNode(root, value):
    if root is None:
        return TreeNode(value)
    if value < root.value:
        root.left = insertNode(root.left, value)
    else:
        root.right = insertNode(root.right, value)
    return root

def inorderTraversal(root, result):
    if root is not None:
        inorderTraversal(root.left, result)
        result.append(root.value)
        inorderTraversal(root.right, result)

def treeSort(arr):
    root = None
    for value in arr:
        root = insertNode(root, value)
    result = []
    inorderTraversal(root, result)
    return result

def ejecutar_ordenamiento_tree():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = treeSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Tree Sort: \n" + str(lista_ordenada))

# Ordenamiento por intercalación (Interpolation Sort)

def interpolationSort(arr):
    n = len(arr)
    if n <= 1:
        return arr

    def insert(arr, i, value):
        j = i - 1
        while j >= 0 and arr[j] > value:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = value

    for i in range(1, n):
        value = arr[i]
        low = 0
        high = i - 1
        while low <= high and value < arr[high]:
            if arr[low] <= value <= arr[high]:
                break
            low += 1
            high -= 1
        if low <= high:
            j = i - 1
            while j >= high:
                arr[j + 1] = arr[j]
                j -= 1
            arr[high] = value
        else:
            insert(arr, i, value)
    return arr

def ejecutar_ordenamiento_interpolation():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = interpolationSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Interpolation Sort: \n" + str(lista_ordenada))

# Ordenamiento de Shell (Shell Sort)

def shellSort(arr):
    n = len(arr)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr

def ejecutar_ordenamiento_shell():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = shellSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Shell Sort: \n" + str(lista_ordenada))

# Ordenamiento de comb (Comb Sort)

def combSort(arr):
    n = len(arr)
    gap = n
    shrink_factor = 1.3
    sorted = False

    while not sorted:
        gap = int(gap / shrink_factor)
        if gap <= 1:
            gap = 1
            sorted = True

        i = 0
        while i + gap < n:
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                sorted = False
            i += 1
    return arr

def ejecutar_ordenamiento_comb():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = combSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Comb Sort: \n" + str(lista_ordenada))

# Ordenamiento de peine (Cocktail Sort)

def cocktailSort(arr):
    n = len(arr)
    swapped = True
    start = 0
    end = n - 1

    while swapped:
        swapped = False

        # Mover el elemento más pequeño al final
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True

        if not swapped:
            break

        swapped = False

        # Mover el elemento más grande al principio
        end -= 1
        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True

        start += 1
    return arr

def ejecutar_ordenamiento_cocktail():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = cocktailSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Cocktail Sort: \n" + str(lista_ordenada))

# Ordenamiento de bote (Gnome Sort)

def gnomeSort(arr):
    n = len(arr)
    i = 0

    while i < n:
        if i == 0 or arr[i] >= arr[i - 1]:
            i += 1
        else:
            arr[i], arr[i - 1] = arr[i - 1], arr[i]
            i -= 1
    return arr

def ejecutar_ordenamiento_gnome():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = gnomeSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Gnome Sort: \n" + str(lista_ordenada))

# Ordenamiento de sacudida (Shaker Sort)

def shakerSort(arr):
    n = len(arr)
    swapped = True
    start = 0
    end = n - 1

    while swapped:
        swapped = False

        # Mover el elemento más grande al final
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True

        if not swapped:
            break

        swapped = False

        # Mover el elemento más pequeño al principio
        end -= 1
        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True

        start += 1
    return arr

def ejecutar_ordenamiento_shaker():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = shakerSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Shaker Sort: \n" + str(lista_ordenada))

# Ordenamiento de pancake (Pancake Sort)

def pancakeSort(arr):
    n = len(arr)

    # Función auxiliar para revertir un subarreglo de arr hasta la posición end
    def reverse(arr, end):
        start = 0
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1

    # Encuentra el índice del elemento máximo en un arreglo dado
    def findMaxIndex(arr, n):
        max_idx = 0
        for i in range(1, n):
            if arr[i] > arr[max_idx]:
                max_idx = i
        return max_idx

    curr_size = n

    # Realizar iteraciones hasta que el tamaño actual del subarreglo sea 1
    while curr_size > 1:
        # Encuentra el índice del elemento máximo en el subarreglo actual
        max_idx = findMaxIndex(arr, curr_size)

        # Voltear el subarreglo desde el inicio hasta el elemento máximo
        reverse(arr, max_idx)

        # Voltear todo el subarreglo desde el inicio hasta el final
        reverse(arr, curr_size - 1)

        curr_size -= 1
    return arr

def ejecutar_ordenamiento_pancake():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = pancakeSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Pancake Sort: \n" + str(lista_ordenada))

# Ordenamiento de cubo (Cube Sort)

def cubeSort(arr):
    # Encontrar el valor máximo y mínimo en el arreglo
    max_value = max(arr)
    min_value = min(arr)

    # Calcular el rango de valores
    value_range = max_value - min_value + 1

    # Crear un arreglo auxiliar para contar las ocurrencias de cada valor
    count = [0] * value_range

    # Contar las ocurrencias de cada valor en el arreglo original
    for num in arr:
        count[num - min_value] += 1

    # Reconstruir el arreglo ordenado utilizando las ocurrencias
    sorted_arr = []
    for i in range(value_range):
        sorted_arr.extend([i + min_value] * count[i])

    return sorted_arr

def ejecutar_ordenamiento_cube():
    # Obtener los datos ingresados por el usuario
    datos = entrada2.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = cubeSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Cube Sort: \n" + str(lista_ordenada))

# Ordenamiento de tim (Tim Sort)

# Tamaño mínimo del subarreglo para aplicar ordenamiento por inserción
MIN_MERGE = 32

def insertionSort(arr, left, right):
    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def merge(arr, left, mid, right):
    len1 = mid - left + 1
    len2 = right - mid
    left_arr = [0] * len1
    right_arr = [0] * len2
    for i in range(len1):
        left_arr[i] = arr[left + i]
    for i in range(len2):
        right_arr[i] = arr[mid + 1 + i]

    i = j = 0
    k = left
    while i < len1 and j < len2:
        if left_arr[i] <= right_arr[j]:
            arr[k] = left_arr[i]
            i += 1
        else:
            arr[k] = right_arr[j]
            j += 1
        k += 1

    while i < len1:
        arr[k] = left_arr[i]
        i += 1
        k += 1

    while j < len2:
        arr[k] = right_arr[j]
        j += 1
        k += 1

def timSort(arr):
    n = len(arr)
    for i in range(0, n, MIN_MERGE):
        insertionSort(arr, i, min(i + MIN_MERGE - 1, n - 1))

    size = MIN_MERGE
    while size < n:
        for left in range(0, n, 2*size):
            mid = left + size - 1
            right = min((left + 2*size - 1), (n-1))
            merge(arr, left, mid, right)
        size *= 2
    return arr

def ejecutar_ordenamiento_tim():
    # Obtener los datos ingresados por el usuario
    datos = entrada.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(int, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = timSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado.config(text="Datos ordenados por metodo Tim Sort: \n" + str(lista_ordenada))

# Ordenamiento de pato (Bucket Sort with duck-typing)

def duckSort(arr):
    # Crear un arreglo de cubetas vacías
    buckets = [[] for _ in range(len(arr))]
    
    # Colocar cada elemento en la cubeta correspondiente
    for value in arr:
        # Ajustar los valores fuera del rango a 0 o 1
        if value < 0:
            value = 0
        elif value > 1:
            value = 1
        
        index = int(value * len(arr))
        buckets[index].append(value)
    
    # Ordenar cada cubeta individualmente
    for bucket in buckets:
        bucket.sort()
    
    # Concatenar las cubetas ordenadas en un solo arreglo
    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(bucket)
    
    return sorted_arr

def ejecutar_ordenamiento_duck():
    # Obtener los datos ingresados por el usuario
    datos = entrada2.get()
    # Separar los datos por comas y convertirlos en una lista de enteros
    lista_datos = list(map(float, datos.split(',')))
    # Aplicar el algoritmo de ordenamiento de burbuja
    lista_ordenada = duckSort(lista_datos)
    # Mostrar los datos ordenados en la ventana
    resultado2.config(text="Datos ordenados por metodo Bucket Sort with duck-typing: \n" + str(lista_ordenada))
    
# Crear la ventana
ventana = tk.Tk()

# Configurar la ventana
ventana.title("Metodos de Ordenamiento")
ventana.geometry("350x680")

# Texto para el usuario
etiqueta = tk.Label(ventana, text="Ingresa los numeros que quieras ordenas,\n seguidos por una coma (,) sin espacios")
etiqueta.pack(pady=8)

# Crear una entrada de texto
entrada = tk.Entry(ventana)
entrada.pack(pady=5)

# Crear un botón
boton1 = tk.Button(ventana, text="Ordenamiento de burbuja (Bubble Sort)", command=ejecutar_ordenamiento_bubble)
boton1.pack(pady=2)

# Crear un botón
boton2 = tk.Button(ventana, text="Ordenamiento por selección (Selection Sort)", command=ejecutar_ordenamiento_sort)
boton2.pack(pady=2)

# Crear un botón
boton3 = tk.Button(ventana, text="Ordenamiento por inserción (Insertion Sort)", command=ejecutar_ordenamiento_insertion)
boton3.pack(pady=2)

# Crear un botón
boton4 = tk.Button(ventana, text="Ordenamiento por mezcla (Merge Sort)", command=ejecutar_ordenamiento_mergeSort)
boton4.pack(pady=2)

# Crear un botón
boton5 = tk.Button(ventana, text="Ordenamiento rápido (Quick Sort)", command=ejecutar_ordenamiento_quick)
boton5.pack(pady=2)

# Crear un botón
boton6 = tk.Button(ventana, text="Ordenamiento por montículos (Heap Sort)", command=ejecutar_ordenamiento_heap)
boton6.pack(pady=2)

# Crear un botón
boton7 = tk.Button(ventana, text="Ordenamiento por cuenta (Counting Sort)", command=ejecutar_ordenamiento_counting)
boton7.pack(pady=2)

# Crear un botón
boton9 = tk.Button(ventana, text="Ordenamiento radix (Radix Sort)", command=ejecutar_ordenamiento_radix)
boton9.pack(pady=2)

# Crear un botón
boton10 = tk.Button(ventana, text="Ordenamiento por árbol (Tree Sort)", command=ejecutar_ordenamiento_tree)
boton10.pack(pady=2)

# Crear un botón
boton11 = tk.Button(ventana, text="Ordenamiento por intercalación (Interpolation Sort)", command=ejecutar_ordenamiento_interpolation)
boton11.pack(pady=2)

# Crear un botón
boton12 = tk.Button(ventana, text="Ordenamiento de Shell (Shell Sort)", command=ejecutar_ordenamiento_shell)
boton12.pack(pady=2)

# Crear un botón
boton13 = tk.Button(ventana, text="Ordenamiento de comb (Comb Sort)", command=ejecutar_ordenamiento_comb)
boton13.pack(pady=2)

# Crear un botón
boton14 = tk.Button(ventana, text="Ordenamiento de peine (Cocktail Sort)", command=ejecutar_ordenamiento_cocktail)
boton14.pack(pady=2)

# Crear un botón
boton15 = tk.Button(ventana, text="Ordenamiento de bote (Gnome Sort)", command=ejecutar_ordenamiento_gnome)
boton15.pack(pady=2)

# Crear un botón
boton16 = tk.Button(ventana, text="Ordenamiento de sacudida (Shaker Sort)", command=ejecutar_ordenamiento_shaker)
boton16.pack(pady=2)

# Crear un botón
boton17 = tk.Button(ventana, text="Ordenamiento de pancake (Pancake Sort)", command=ejecutar_ordenamiento_pancake)
boton17.pack(pady=2)

# Crear un botón
boton18 = tk.Button(ventana, text="Ordenamiento de cubo (Cube Sort)", command=ejecutar_ordenamiento_cube)
boton18.pack(pady=2)

# Crear un botón
boton19 = tk.Button(ventana, text="Ordenamiento de tim (Tim Sort)", command=ejecutar_ordenamiento_tim)
boton19.pack(pady=2)

# Crear una etiqueta para mostrar el resultado
resultado = tk.Label(ventana)
resultado.pack(pady=10)

# Crear la ventana
ventana2 = tk.Tk()

# Configurar la ventana
ventana2.title("Metodos de Ordenamiento")
ventana2.geometry("350x240")

# Texto para el usuario
etiqueta2 = tk.Label(ventana2, text="Ingresa los numeros que quieras ordenas,\n seguidos por una coma (,) sin espacios\n estos metodos de ordenamiento solo pueden utilizar\n valores dentro del rango de 0 a 1")
etiqueta2.pack(pady=8)

# Crear una entrada de texto
entrada2 = tk.Entry(ventana2)
entrada2.pack(pady=5)

# Crear un botón
boton8 = tk.Button(ventana2, text="Ordenamiento de cubetas (Bucket Sort)", command=ejecutar_ordenamiento_bucket)
boton8.pack(pady=2)

# Crear un botón
boton20 = tk.Button(ventana2, text="Ordenamiento de pato (Bucket Sort with duck-typing)", command=ejecutar_ordenamiento_duck)
boton20.pack(pady=2)

# Crear una etiqueta para mostrar el resultado
resultado2 = tk.Label(ventana2)
resultado2.pack(pady=10)

# Iniciar el bucle de eventos de la ventana
ventana.mainloop()
ventana2.mainloop()