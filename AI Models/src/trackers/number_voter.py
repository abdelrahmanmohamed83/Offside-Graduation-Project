from collections import Counter

class NumberVotingSystem:
    def __init__(self, required_frames=None):
        """
        نظام تصويت ديناميكي مستمر يعتمد على القيمة الأكثر تكراراً (Mode) عبر الفيديو بالكامل.
        """
        self.required_frames = required_frames
        self.history = {}  # track_id -> قائمة بجميع القراءات المرصودة

    @property
    def final_numbers(self):
        """إرجاع الرقم الأكثر تكراراً لكل لاعب بشكل ديناميكي للتوافق."""
        res = {}
        for tid, votes in self.history.items():
            if votes:
                most_common = Counter(votes).most_common(1)[0][0]
                if most_common:
                    res[tid] = most_common
        return res

    def update(self, track_id, predicted_number):
        """يستقبل القراءة الجديدة ويحدث التصويت، ثم يرجع الرقم الفائز حتى الآن."""
        if track_id not in self.history:
            self.history[track_id] = []
            
        if predicted_number is not None and predicted_number != "":
            self.history[track_id].append(predicted_number)
            
        if self.history[track_id]:
            return Counter(self.history[track_id]).most_common(1)[0][0]
            
        return ""