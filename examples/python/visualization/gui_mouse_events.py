#!/usr/bin/env python3
"""
Example: MouseEvent creation and injection

Demonstrates:
- Creating MouseEvent objects using static methods
- Injecting events into GUI windows  
- Receiving events via callbacks

Note: This is an interactive example, not a unit test.
"""

import open3d as o3d
import open3d.visualization.gui as gui
import threading
import time


def main():
    """Demonstrate MouseEvent functionality."""
    
    app = gui.Application.instance
    app.initialize()
    
    window = app.create_window("MouseEvent Demo", 600, 400)
    
    # Event tracking with specific validation
    received_events = []
    
    # UI
    layout = gui.Vert(10, gui.Margins(20))
    status = gui.Label("Starting automated demo...")
    results = gui.Label("Events: None")
    layout.add_child(status)
    layout.add_child(results)
    window.add_child(layout)
    
    def on_mouse(event):
        """Log received events with validation."""
        type_str = {
            gui.MouseEvent.Type.BUTTON_DOWN: "DOWN",
            gui.MouseEvent.Type.BUTTON_UP: "UP", 
            gui.MouseEvent.Type.MOVE: "MOVE",
            gui.MouseEvent.Type.WHEEL: "WHEEL"
        }.get(event.type, "UNKNOWN")
        
        event_str = f"{type_str} at ({event.x},{event.y})"
        
        received_events.append(event)
        results.text = f"Events: {len(received_events)} | Last: {event_str}"
        
        # Validate event structure
        assert hasattr(event, 'type')
        assert hasattr(event, 'x')
        assert hasattr(event, 'y')
        
        return False
    
    window.set_on_mouse(on_mouse)
    
    def demo():
        """Inject test events with specific validation."""
        time.sleep(1)
        
        # Test sequence: move, click, move, wheel
        events = [
            gui.MouseEvent.move(100, 100),
            gui.MouseEvent.button_down(100, 100, gui.MouseButton.LEFT, 0),
            gui.MouseEvent.button_up(100, 100, gui.MouseButton.LEFT, 0),
            gui.MouseEvent.wheel(150, 150, 0, 1, 0)
        ]
        
        for i, event in enumerate(events):
            status.text = f"Injecting event {i+1}/4..."
            window.post_mouse_event(event)
            time.sleep(0.5)
        
        # Verify we got expected events
        time.sleep(0.5)
        if len(received_events) >= 4:
            status.text = "✅ Demo completed successfully"
            
            # Specific validation instead of just count
            assert received_events[0].type == gui.MouseEvent.Type.MOVE
            assert received_events[0].x == 100
            assert received_events[1].type == gui.MouseEvent.Type.BUTTON_DOWN
            assert received_events[2].type == gui.MouseEvent.Type.BUTTON_UP
            assert received_events[3].type == gui.MouseEvent.Type.WHEEL
            
        else:
            status.text = f"❌ Expected 4+ events, got {len(received_events)}"
    
    threading.Thread(target=demo, daemon=True).start()
    app.run()


if __name__ == "__main__":
    main()