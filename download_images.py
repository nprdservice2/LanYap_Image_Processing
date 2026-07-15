import os
import csv
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

def sanitize_folder_name(name):
    # Remove characters that are invalid in folder names on Windows/Linux
    invalid_chars = '<>:"/\\|?*'
    sanitized = ''.join(c for c in name if c not in invalid_chars)
    return sanitized.strip()

def get_jpg_url(url):
    # Parse url to replace path extension with .jpg for Cloudinary to convert on-the-fly
    parsed = urllib.parse.urlparse(url)
    path = parsed.path
    root, _ = os.path.splitext(path)
    new_path = root + '.jpg'
    new_parsed = parsed._replace(path=new_path)
    return urllib.parse.urlunparse(new_parsed)

def download_image(url, filepath):
    try:
        # User-Agent to prevent getting blocked by basic anti-scraping checks
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            with open(filepath, 'wb') as f:
                f.write(response.read())
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    csv_file = 'lanyap26June.csv'
    output_dir = 'downloaded_images'
    max_images_per_class = 1000
    
    print(f"Reading CSV file '{csv_file}'...")
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found.")
        return

    # Group URLs by sanitized class name
    class_to_urls = defaultdict(list)
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("Error: CSV file is empty.")
            return
            
        for row_idx, row in enumerate(reader, start=2):
            if len(row) < 2:
                continue
            name, url = row[0], row[1]
            if not url or not url.startswith('http'):
                continue
            
            sanitized_name = sanitize_folder_name(name)
            if not sanitized_name:
                sanitized_name = "Unclassified"
                
            class_to_urls[sanitized_name].append(url)

    print(f"Found {len(class_to_urls)} unique classes in the CSV.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    def process_class(class_name, urls):
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        success_count = 0
        url_idx = 0
        total_urls = len(urls)
        
        while success_count < max_images_per_class and url_idx < total_urls:
            original_url = urls[url_idx]
            # Convert URL to fetch JPG format from Cloudinary
            url = get_jpg_url(original_url)
            # Use unique filename format: <CategoryName>_<Index>.jpg
            filename = f"{class_name}_{success_count + 1:02d}.jpg"
            filepath = os.path.join(class_dir, filename)
            
            # Check if file already exists
            if os.path.exists(filepath):
                success_count += 1
                url_idx += 1
                continue
                
            success, err = download_image(url, filepath)
            if success:
                success_count += 1
            else:
                print(f"  [{class_name}] Failed to download URL index {url_idx} ({url}). Error: {err}")
            url_idx += 1
            
        return class_name, success_count

    max_workers = 16  # Adjust based on bandwidth and performance
    print(f"Starting downloads (converting to JPG with unique filenames) using up to {max_workers} concurrent threads...")
    
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_class, name, urls): name for name, urls in class_to_urls.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                class_name, count = future.result()
                results[class_name] = count
                print(f"Completed class: '{class_name}' - downloaded {count} images.")
            except Exception as e:
                print(f"Error processing class '{name}': {e}")
                
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY (JPG with Unique Names)")
    print("="*50)
    for class_name, count in sorted(results.items()):
        print(f"  {class_name:<35}: {count} images")
    print("="*50)
    print(f"Total downloaded images are stored in the '{output_dir}' directory.")

if __name__ == '__main__':
    main()