import gzip

# Path to the .gz file
gz_file_path = '../eye_tracking_participant/20241030T091636Z/gazedata.gz'

# Open the .gz file in text mode
with gzip.open(gz_file_path, 'rt') as f:
    content = f.read()
    print(content)