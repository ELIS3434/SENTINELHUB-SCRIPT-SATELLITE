# Sentinel Hub Image Viewer

A Python application for viewing and processing Sentinel-2 satellite imagery using the Sentinel Hub API.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Sentinel Hub Configuration

1. Create a Sentinel Hub account at [https://www.sentinel-hub.com/](https://www.sentinel-hub.com/)
2. Go to your account settings and create OAuth credentials
3. Create a `.env` file in the project root with the following content:
```env
SENTINEL_HUB_CLIENT_ID=your_client_id_here
SENTINEL_HUB_CLIENT_SECRET=your_client_secret_here
```

### 3. Environment Variables
Create a `.env` file in the project root directory with your Sentinel Hub credentials:
```env
SENTINEL_HUB_CLIENT_ID=your_client_id_here
SENTINEL_HUB_CLIENT_SECRET=your_client_secret_here
```

## Features

- High-resolution Sentinel-2 imagery display
- Enhanced visualization using 10m resolution bands
- Vegetation enhancement using NDVI
- Interactive location selection
- Customizable image quality settings

## Usage

1. Run the application:
```bash
python image.py
```

2. The application will prompt you to:
   - Enter a location name
   - Select image quality settings
   - Set update interval

3. Controls:
   - Mouse wheel: Zoom in/out
   - Left click + drag: Pan the image
   - ESC: Exit the application

## Image Processing

The application uses Sentinel-2's high-resolution bands:
- B02 (Blue)
- B03 (Green)
- B04 (Red)
- B08 (NIR)

Image enhancement includes:
- Contrast stretching
- Vegetation enhancement using NDVI
- Brightness adjustment
- Color balancing

## Troubleshooting

1. If you see no image:
   - Check your Sentinel Hub credentials
   - Verify internet connection
   - Try a different location
   - Adjust the date range if necessary

2. If the image is too dark/bright:
   - The evalscript in image.py can be adjusted
   - Modify the gain value (currently 2.5)
   - Adjust contrast stretching parameters

## Dependencies

See `requirements.txt` for a complete list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
