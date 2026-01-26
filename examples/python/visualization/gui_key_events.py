#!/usr/bin/env python3
"""
Example: KeyEvent creation and injection

Demonstrates:
- Creating KeyEvent objects
- Injecting events into GUI windows  
- Receiving events via callbacks

Note: This is an interactive example, not a unit test.
"""

import open3d as o3d
import open3d.visualization.gui as gui
import threading
import time


def main():
    """Demonstrate KeyEvent functionality."""
    
    app = gui.Application.instance
    app.initialize()
    
    window = app.create_window("KeyEvent Demo", 600, 400)
    
    # Event tracking with specific validation
    received_events = []
    
    # UI
    layout = gui.Vert(10, gui.Margins(20))
    status = gui.Label("Starting automated demo...")
    results = gui.Label("Events: None")
    layout.add_child(status)
    layout.add_child(results)
    window.add_child(layout)
    
    def on_key(event):
        """Log received events with validation."""
        event_str = f"{chr(event.key) if 32 <= event.key <= 126 else event.key}:"
        event_str += f"{'DOWN' if event.type == gui.KeyEvent.Type.DOWN else 'UP'}"
        
        received_events.append(event)
        results.text = f"Events: {len(received_events)} | Last: {event_str}"
        
        # Validate event structure
        assert hasattr(event, 'key')
        assert hasattr(event, 'type') 
        assert hasattr(event, 'is_repeat')
        
        return False
    
    window.set_on_key(on_key)
    
    def demo():
        """Inject test events with specific validation."""
        time.sleep(1)
        
        # Test sequence: A down, A up, Space down, Space up
        events = [
            gui.KeyEvent(gui.KeyEvent.Type.DOWN, gui.KeyName.A, False),
            gui.KeyEvent(gui.KeyEvent.Type.UP, gui.KeyName.A, False),
            gui.KeyEvent(gui.KeyEvent.Type.DOWN, gui.KeyName.SPACE, False),
            gui.KeyEvent(gui.KeyEvent.Type.UP, gui.KeyName.SPACE, False),
        ]
        
        for i, event in enumerate(events):
            status.text = f"Injecting event {i+1}/4..."
            window.post_key_event(event)
            time.sleep(0.5)
        
        # Verify we got expected events
        time.sleep(0.5)
        if len(received_events) >= 4:
            status.text = "✅ Demo completed successfully"
            
            # Specific validation instead of just count
            assert received_events[0].key == gui.KeyName.A
            assert received_events[0].type == gui.KeyEvent.Type.DOWN
            assert received_events[1].type == gui.KeyEvent.Type.UP
            assert received_events[2].key == gui.KeyName.SPACE
            
        else:
            status.text = f"❌ Expected 4+ events, got {len(received_events)}"
    
    threading.Thread(target=demo, daemon=True).start()
    app.run()


if __name__ == "__main__":
    main()