import numpy as np
import cv2

# --- Konfigurasi Perlin Noise ---
WIDTH, HEIGHT = 512, 512 # Ukuran gambar output
OCTAVES = 6             # Jumlah lapisan noise
PERSISTENCE = 0.5       # Pengurangan amplitudo di setiap octave
LACUNARITY = 2.0        # Peningkatan frekuensi di setiap octave
SCALE = 0.01            # Skala awal noise (semakin kecil, semakin 'zoom out')
SEED = 42               # Seed untuk reproduktifitas

# --- 1. Vektor Gradien Tetap (Pre-defined) ---
# Menggunakan 8 vektor gradien untuk Perlin Noise 2D
GRADIENTS_2D = np.array([
    [1, 1], [-1, 1], [1, -1], [-1, -1],
    [1, 0], [-1, 0], [0, 1], [0, -1]
], dtype=np.float32) # Penting: Gunakan float32 untuk perhitungan

# --- Fungsi Utility ---

def create_permutation_table(seed):
    """
    Membuat dan mengacak tabel permutasi berdasarkan seed.
    Menggandakan untuk wrapping yang mulus.
    """
    np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    return np.concatenate((p, p))

# Panggil sekali untuk mendapatkan tabel permutasi global
P = create_permutation_table(SEED)

def get_gradient_vector(x_grid, y_grid):
    """
    Mengambil vektor gradien pseudo-random untuk titik grid integer (x_grid, y_grid)
    menggunakan tabel permutasi P.
    """
    # Pastikan koordinat adalah integer
    X = int(x_grid)
    Y = int(y_grid)

    # Hashing menggunakan tabel permutasi
    idx_x = X % 256
    idx_y = Y % 256
    
    # Perlin's hash function (modifikasi ringan dari versi asli untuk kesederhanaan)
    # Ini mencari nilai dari P berdasarkan kombinasi x dan y
    hash_val = P[P[idx_x] + idx_y]

    # Pilih salah satu dari 8 vektor gradien
    gradient_index = hash_val % len(GRADIENTS_2D)
    return GRADIENTS_2D[gradient_index]

def fade(t):
    """
    Fungsi smoothing Perlin (smootherstep) untuk interpolasi yang mulus.
    f(t) = 6t^5 - 15t^4 + 10t^3
    """
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def lerp(a, b, t):
    """
    Interpolasi linier antara a dan b berdasarkan t.
    """
    return a + t * (b - a)

# --- Fungsi Perlin Noise Utama ---

def perlin_noise(x, y):
    """
    Menghitung nilai Perlin Noise pada koordinat floating-point (x, y).
    """
    # 1. Tentukan Titik Grid Terdekat
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1

    # 2. Hitung Vektor Jarak ke 4 Sudut Grid
    dx0 = x - x0
    dy0 = y - y0
    dx1 = x - x1 # atau x - (x0 + 1)
    dy1 = y - y1 # atau y - (y0 + 1)

    # Vektor jarak dari (x,y) ke 4 sudut
    vec00 = np.array([dx0, dy0], dtype=np.float32)
    vec10 = np.array([dx1, dy0], dtype=np.float32)
    vec01 = np.array([dx0, dy1], dtype=np.float32)
    vec11 = np.array([dx1, dy1], dtype=np.float32)
    
    # 3. Dapatkan Vektor Gradien untuk Setiap Sudut
    g00 = get_gradient_vector(x0, y0)
    g10 = get_gradient_vector(x1, y0)
    g01 = get_gradient_vector(x0, y1)
    g11 = get_gradient_vector(x1, y1)

    # 4. Hitung Dot Product (Produk Titik) untuk Setiap Sudut
    # (gradien_vektor . vektor_jarak)
    n00 = np.dot(g00, vec00)
    n10 = np.dot(g10, vec10)
    n01 = np.dot(g01, vec01)
    n11 = np.dot(g11, vec11)

    # 5. Lakukan Smoothing pada Vektor Jarak Fraksional
    # Ini adalah 't' untuk fungsi lerp, nilai antara 0 dan 1
    sx = fade(dx0) 
    sy = fade(dy0)

    # 6. Interpolasi (Lerp)
    # Interpolasi horizontal di baris atas
    ix0 = lerp(n00, n10, sx)
    # Interpolasi horizontal di baris bawah
    ix1 = lerp(n01, n11, sx)
    # Interpolasi vertikal hasil dari kedua interpolasi horizontal
    final_value = lerp(ix0, ix1, sy)

    return final_value

# --- Generate Gambar Perlin Noise ---

def generate_perlin_image(width, height, octaves, persistence, lacunarity, scale, seed):
    noise_map = np.zeros((height, width), dtype=np.float32)
    
    # Panggil ulang tabel permutasi dengan seed untuk octave (penting jika Anda ingin setiap octave berbeda seed)
    # Dalam implementasi ini, kita menggunakan seed global yang sama untuk semua octave
    # Tapi frekuensi dan amplitudo yang berbeda
    
    max_amplitude = 0 # Digunakan untuk normalisasi nanti
    amplitude = 1
    frequency = scale

    for _ in range(octaves):
        for y in range(height):
            for x in range(width):
                # Koordinat yang diskalakan untuk octave saat ini
                sample_x = x * frequency
                sample_y = y * frequency
                
                # Tambahkan noise dari octave saat ini, dikalikan dengan amplitudonya
                noise_map[y, x] += perlin_noise(sample_x, sample_y) * amplitude
        
        # Perbarui amplitude dan frequency untuk octave berikutnya
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    
    # Normalisasi noise_map ke rentang 0-1
    # Kita perlu mencari min dan max noise_map untuk normalisasi yang benar
    # Perlin noise asli menghasilkan nilai antara sekitar -1 dan 1
    # Tapi dengan octaves, range bisa lebih besar
    min_val = noise_map.min()
    max_val = noise_map.max()
    
    if max_val - min_val == 0: # Hindari pembagian nol jika semua nilai sama
        normalized_noise = np.zeros((height, width), dtype=np.float32)
    else:
        normalized_noise = (noise_map - min_val) / (max_val - min_val)
    
    return normalized_noise

# --- Proses Utama ---
if __name__ == "__main__":
    print("Mulai membuat gambar Perlin Noise...")
    # Generate noise map
    image_data = generate_perlin_image(WIDTH, HEIGHT, OCTAVES, PERSISTENCE, LACUNARITY, SCALE, SEED)
    
    # Konversi ke gambar 8-bit (0-255) untuk OpenCV
    # Kalikan dengan 255 dan konversi ke tipe unsigned 8-bit integer
    image_display = (image_data * 255).astype(np.uint8)
    
    # Tampilkan gambar menggunakan OpenCV
    cv2.imshow("Perlin Noise Image", image_display)
    
    # Simpan gambar (opsional)
    cv2.imwrite("perlin.png", image_display)
    
    print("Tekan tombol apapun untuk menutup gambar...")
    cv2.waitKey(0) # Menunggu user menekan tombol
    cv2.destroyAllWindows() # Menutup semua window OpenCV