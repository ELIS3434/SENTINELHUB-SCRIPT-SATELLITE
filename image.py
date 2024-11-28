import os
import cv2
import numpy as np
from datetime import datetime, timedelta
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, BBox, CRS, MimeType
from dotenv import load_dotenv
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Load environment variables
load_dotenv()

# Configure Sentinel Hub
config = SHConfig()
config.sh_client_id = os.getenv('SENTINEL_HUB_CLIENT_ID')
config.sh_client_secret = os.getenv('SENTINEL_HUB_CLIENT_SECRET')

if not config.sh_client_id or not config.sh_client_secret:
    print("Please provide the credentials (client ID and client secret).")
    exit()

# Define the request with enhanced settings
evalscript = """
//VERSION=3

function setup() {
    return {
        input: [{
            bands: ["B02", "B03", "B04", "B08"],
            units: "DN"
        }],
        output: {
            bands: 3,
            sampleType: "AUTO"
        }
    };
}

function evaluatePixel(sample) {
    // Get band values and normalize them
    let B02 = sample.B02 / 10000;  // Blue
    let B03 = sample.B03 / 10000;  // Green
    let B04 = sample.B04 / 10000;  // Red
    let B08 = sample.B08 / 10000;  // NIR

    // Enhanced natural color
    let gain = 2.5;  // Increase brightness
    let R = gain * B04;
    let G = gain * B03;
    let B = gain * B02;

    // Apply contrast stretching
    let stretchMin = 0.0;
    let stretchMax = 0.8;
    R = (R - stretchMin) / (stretchMax - stretchMin);
    G = (G - stretchMin) / (stretchMax - stretchMin);
    B = (B - stretchMin) / (stretchMax - stretchMin);

    // Enhance vegetation
    let NDVI = (B08 - B04) / (B08 + B04);
    let vegetation_enhancement = Math.max(0, NDVI) * 0.2;
    
    // Apply vegetation enhancement
    G = G * (1 + vegetation_enhancement);

    // Final adjustments and clipping
    return [
        Math.min(1, Math.max(0, R)),
        Math.min(1, Math.max(0, G)),
        Math.min(1, Math.max(0, B))
    ];
}
"""


class LocationManager:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="my_satellite_viewer")
        self.current_location = None
        self.current_bbox = None
    
    def update_bbox_from_location(self, location_name, span=0.1):
        try:
            location = self.geolocator.geocode(location_name)
            if location:
                self.current_bbox = BBox(
                    bbox=[
                        location.longitude - span/2,
                        location.latitude - span/2,
                        location.longitude + span/2,
                        location.latitude + span/2
                    ],
                    crs=CRS.WGS84
                )
                self.current_location = location.address
                return True
            return False
        except Exception as e:
            print(f"❌ Error updating location: {str(e)}")
            return False

class ImageViewer:
    def __init__(self, window_name="Live Satellite View"):
        self.window_name = window_name
        self.zoom_factor = 1.0
        self.max_zoom = 5.0  # Increased max zoom to 5x
        self.min_zoom = 0.2
        self.pan_x = 0
        self.pan_y = 0
        self.dragging = False
        self.last_mouse_pos = None
        self.show_coordinates = True
        self.mouse_lat = None
        self.mouse_lon = None
        self.current_bbox = None
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
    
    def update_bbox(self, bbox):
        self.current_bbox = bbox
    
    def pixel_to_coordinates(self, x, y, img_width, img_height):
        if self.current_bbox is None:
            return None, None
        
        # Convert pixel coordinates to relative position
        rel_x = x / img_width
        rel_y = 1 - (y / img_height)  # Invert Y axis
        
        # Calculate actual coordinates
        lon = self.current_bbox.lower_left[0] + rel_x * (self.current_bbox.upper_right[0] - self.current_bbox.lower_left[0])
        lat = self.current_bbox.lower_left[1] + rel_y * (self.current_bbox.upper_right[1] - self.current_bbox.lower_left[1])
        
        return lat, lon
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:  # Scroll up
                self.zoom_factor = min(self.max_zoom, self.zoom_factor * 1.1)
            else:  # Scroll down
                self.zoom_factor = max(self.min_zoom, self.zoom_factor / 1.1)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]
                self.pan_x += dx
                self.pan_y += dy
                self.last_mouse_pos = (x, y)
            
            # Update mouse coordinates
            if self.current_bbox is not None:
                h, w = param[0].shape[:2] if param and len(param) > 0 else (512, 512)
                self.mouse_lat, self.mouse_lon = self.pixel_to_coordinates(x, y, w, h)
    
    def process_key(self, key):
        if key == ord('+') or key == ord('='):
            self.zoom_factor = min(self.max_zoom, self.zoom_factor * 1.1)
        elif key == ord('-') or key == ord('_'):
            self.zoom_factor = max(self.min_zoom, self.zoom_factor / 1.1)
        elif key == ord('c'):
            self.show_coordinates = not self.show_coordinates
        return key == ord('q')
    
    def display_image(self, image):
        if image is None:
            return
        
        # Get the original image dimensions
        h, w = image.shape[:2]
        
        # Calculate the new size based on zoom
        new_w = int(w * self.zoom_factor)
        new_h = int(h * self.zoom_factor)
        
        # Resize the image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create a black canvas larger than the image
        canvas_w = max(w, new_w) + 200
        canvas_h = max(h, new_h) + 200
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Calculate center position with panning
        x_offset = (canvas_w - new_w) // 2 + self.pan_x
        y_offset = (canvas_h - new_h) // 2 + self.pan_y
        
        # Place the resized image on the canvas
        try:
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        except ValueError:
            self.pan_x = 0
            self.pan_y = 0
            return
        
        # Add information overlay
        info_text = [
            f"Zoom: {self.zoom_factor:.2f}x | Use +/- or mouse wheel to zoom | Left click and drag to pan",
            "Press 'c' to toggle coordinate display | 'q' to quit"
        ]
        
        if self.show_coordinates and self.mouse_lat is not None and self.mouse_lon is not None:
            info_text.append(f"Coordinates: {self.mouse_lat:.6f}°N, {self.mouse_lon:.6f}°E")
        
        for i, text in enumerate(info_text):
            cv2.putText(canvas, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the canvas
        cv2.imshow(self.window_name, canvas)

class ImageQualitySettings:
    def __init__(self):
        self.quality_preset = "HIGH"  # LOW, MEDIUM, HIGH, ULTRA
        self.max_cloud_coverage = 30  # Default max cloud coverage percentage
        
    def get_size(self):
        if self.quality_preset == "LOW":
            return [1024, 1024]
        elif self.quality_preset == "MEDIUM":
            return [1800, 1800]
        elif self.quality_preset == "HIGH":
            return [2500, 2500]
        else:  # ULTRA
            return [2500, 2500]  # Maximum allowed by API
            
    def get_upsampling(self):
        if self.quality_preset == "LOW":
            return "NEAREST"
        else:
            return "BICUBIC"

def get_user_location():
    print("\n" + "="*50)
    print("🌍 Welcome to Live Satellite View!")
    print("="*50 + "\n")
    
    while True:
        print("📍 Enter a location (city, country or address):")
        print("   Example: 'Paris, France' or '123 Main St, New York'")
        location = input("➤ ").strip()
        
        if not location:
            print("❌ Location cannot be empty. Please try again.\n")
            continue
            
        try:
            geolocator = Nominatim(user_agent="my_satellite_viewer")
            location_data = geolocator.geocode(location)
            
            if location_data is None:
                print("❌ Location not found. Please try again with a different location.\n")
                continue
                
            print(f"\n✅ Found: {location_data.address}")
            print(f"   Coordinates: {location_data.latitude:.4f}°N, {location_data.longitude:.4f}°E\n")
            
            return location_data
            
        except GeocoderTimedOut:
            print("⚠️ Request timed out. Please try again.\n")
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")

def get_user_quality_settings():
    print("\n" + "="*50)
    print("🎨 Image Quality Settings")
    print("="*50)
    
    quality_settings = ImageQualitySettings()
    
    print("\nSelect image quality:")
    print("1) 🔷 Low     - 1024x1024, faster loading")
    print("2) 🔶 Medium  - 1800x1800, balanced")
    print("3) 💎 High    - 2500x2500, detailed")
    print("4) ✨ Ultra   - 2500x2500, maximum quality\n")
    
    while True:
        choice = input("Enter your choice (1-4) ➤ ").strip()
        if choice == "1":
            quality_settings.quality_preset = "LOW"
            print("\n✅ Selected: Low quality")
        elif choice == "2":
            quality_settings.quality_preset = "MEDIUM"
            print("\n✅ Selected: Medium quality")
        elif choice == "3":
            quality_settings.quality_preset = "HIGH"
            print("\n✅ Selected: High quality")
        elif choice == "4":
            quality_settings.quality_preset = "ULTRA"
            print("\n✅ Selected: Ultra quality")
        else:
            print("❌ Invalid choice. Please enter 1-4.\n")
            continue
            
        print(f"   Resolution: {quality_settings.get_size()[0]}x{quality_settings.get_size()[1]}")
        break
    
    return quality_settings

def get_update_interval():
    print("\n" + "="*50)
    print("⏱️  Update Interval Settings")
    print("="*50)
    
    while True:
        print("\nEnter the update interval in seconds (10-3600):")
        print("Recommended: 60 for frequent updates, 300 for balanced, 900 for less frequent")
        try:
            interval = int(input("➤ ").strip())
            if 10 <= interval <= 3600:
                print(f"\n✅ Update interval set to: {interval} seconds")
                return interval
            else:
                print("❌ Please enter a value between 10 and 3600 seconds")
        except ValueError:
            print("❌ Please enter a valid number")

def setup_viewer():
    print("\n" + "="*50)
    print("🎮 Controls Guide")
    print("="*50)
    print("\n🖱️  Mouse Controls:")
    print("   • Scroll Wheel     - Zoom in/out (up to 5x)")
    print("   • Left Click+Drag  - Pan image")
    print("   • Right Click      - Show coordinates")
    print("\n⌨️  Keyboard Controls:")
    print("   • R - Refresh image")
    print("   • C - Toggle coordinate display")
    print("   • Q - Quit application")
    print("=" * 40)
    
    print("\nViewer Controls:")
    print("--------------")
    print("- Mouse wheel or +/- : Zoom in/out")
    print("- Left click and drag: Pan the image")
    print("- 'c' key: Toggle coordinate display")
    print("- 'l' key: Change location")
    print("- 'q' key: Quit the viewer")
    print("\n" + "="*50 + "\n")
    
    location_data = get_user_location()
    location_manager = LocationManager()
    
    if not location_manager.update_bbox_from_location(location_data.address):
        print("❌ Failed to set up location. Please try again.")
        exit()
        
    # Show coordinates after location selection
    print("\n📍 Location Details:")
    print(f"   Address: {location_manager.current_location}")
    print(f"   Coordinates: {location_manager.current_bbox.lower_left[1]:.4f}°N, {location_manager.current_bbox.lower_left[0]:.4f}°E to")
    print(f"               {location_manager.current_bbox.upper_right[1]:.4f}°N, {location_manager.current_bbox.upper_right[0]:.4f}°E")
    
    proceed = input("\nPress Enter to continue with this location (or 'q' to quit): ").strip().lower()
    if proceed == 'q':
        print("Exiting...")
        exit()
    
    quality_settings = get_user_quality_settings()
    update_interval = get_update_interval()
    
    return location_manager, quality_settings, update_interval

def get_current_image(bbox, quality_settings):
    current_date = datetime.now()
    start_date = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")  # Increased search window
    end_date = current_date.strftime("%Y-%m-%d")
    
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(start_date, end_date),
                mosaicking_order="leastCC",  # Least cloud coverage
                upsampling=quality_settings.get_upsampling(),
                downsampling="BICUBIC"
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.PNG)
        ],
        bbox=bbox,
        size=quality_settings.get_size(),
        config=config
    )
    
    try:
        image = request.get_data()[0]
        if image is None:
            print("⚠️ No clear images found in the last 30 days. Try increasing cloud coverage limit.")
            return None
        return image
    except Exception as e:
        print(f"❌ Error fetching image: {e}")
        return None

def main():
    print("\n🛰️ Welcome to Satellite Image Viewer!")
    print("="*40)
    
    # Setup viewer with user input
    try:
        location_manager, quality_settings, update_interval = setup_viewer()
    except Exception as e:
        print(f"Error during setup: {e}")
        print("Exiting...")
        exit()
    
    # Initialize viewer
    viewer = ImageViewer()
    viewer.update_bbox(location_manager.current_bbox)
    
    last_update = 0
    
    print("\n🎮 Viewer Controls:")
    print("-" * 20)
    print("🖱️ Mouse wheel or +/- : Zoom in/out")
    print("🖱️ Left click and drag: Pan the image")
    print("🖱️ Right click: Show coordinates")
    print("⌨️ 'R' key: Refresh image")
    print("⌨️ 'Q' key: Quit")
    print("=" * 40)
    
    print("\nViewer Controls:")
    print("--------------")
    print("- Mouse wheel or +/- : Zoom in/out")
    print("- Left click and drag: Pan the image")
    print("- 'c' key: Toggle coordinate display")
    print("- 'l' key: Change location")
    print("- 'q' key: Quit the viewer")
    print("\nLoading satellite imagery...")
    
    while True:
        try:
            # Check for location input
            if cv2.waitKey(1) & 0xFF == ord('l'):
                new_location = input("\nEnter city name (or press Enter to cancel): ").strip()
                if new_location:
                    if location_manager.update_bbox_from_location(new_location):
                        viewer.update_bbox(location_manager.current_bbox)
                        print(f"Location updated to: {new_location}")
                        last_update = 0  # Force image update
            
            current_time = time.time()
            
            # Update image every interval
            if current_time - last_update >= update_interval:
                image = get_current_image(location_manager.current_bbox, quality_settings)
                if image is not None:
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    last_update = current_time
            
            # Display current image with viewer
            if 'image_bgr' in locals():
                viewer.display_image(image_bgr)
            
            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF
            if viewer.process_key(key):
                break
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(5)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()