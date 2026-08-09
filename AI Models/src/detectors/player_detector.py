from ultralytics import YOLO
from src.config import PLAYER_CONFIDENCE

class PlayerDetector:
    def __init__(self, weights_path):
        print("Loading player detection model (with tracking)...")
        self.model = YOLO(weights_path)

    def detect(self, frame):
        """
        Detects and tracks players in the frame.
        """
        results = self.model.track(
            frame, 
            persist=True, 
            tracker="botsort.yaml", 
            conf=PLAYER_CONFIDENCE, 
            classes=[0],
            verbose=False
        )
        
        tracked_players = []
        for result in results:
            boxes = result.boxes
            # التأكد من وجود أرقام تتبع (IDs)
            if boxes.id is not None: 
                track_ids = boxes.id.int().cpu().tolist()
                bboxes = boxes.xyxy.int().cpu().tolist()
                
                for track_id, bbox in zip(track_ids, bboxes):
                    tracked_players.append((track_id, bbox))
                    
        return tracked_players