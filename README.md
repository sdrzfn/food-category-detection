# Food Detection Category ✨
# "Implementing Food Detection Using Computer Vision"


## Deskripsi

Proyek ini bertujuan untuk mengetes model computer vision dalam mengenali dan mendeteksi berbagai macam kategori makanan. Disini kami menggunakan sementara dengan 10 jenis saja


## Struktur Direktori

- **/train_set**: Direktori ini berisi dataset berupa gambar yang digunakan untuk training model.
- **bing-image.downloader.ipynb**: File yang digunakan untuk scraping dataset dari internet menggunakan library dari bing.
- **Food_Image_Classifier.ipynb**: File ini yang digunakan untuk membangun model computer vision dan melatihnya dengan datasets yang telah disediakan
- **food-detection-app.py**: File ini yang digunakan untuk mengatur tampilan di streamlit yang berperan sebagai deployment model.


## Setup Environment dan instalasi
```
conda create --name main-ds python=3.9
conda activate main-ds
pip install streamlit
pip install -r requirements.txt
```


## Penggunaan
1. Masuk ke direktori proyek (Local):

    ```shell
    cd food-detection
    streamlit run food-detection-app.py
    ```

    Jika mengalami kendala, bisa menggunakan command ini:

    ```shell
    py -m streamlit run food-detection-app.py
    ```
    Atau bisa dengan kunjungi website ini [Project Data Analytics](http://192.168.190.239:8501)




#
