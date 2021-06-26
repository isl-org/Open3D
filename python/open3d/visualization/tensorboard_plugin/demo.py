import tensorflow as tf
from open3d.visualization.tensorboard_plugin import summary


def main():
    writer = tf.summary.create_file_writer("demo_logs")
    with writer.as_default():
        summary.greeting(
            "guestbook",
            "Alice",
            step=0,
            description="Sign your name!",
        )
        summary.greeting("guestbook", "Bob",
                         step=1)  # no need for `description`
        summary.greeting("guestbook", "Cheryl", step=2)
        summary.greeting("more_names", "David", step=4)


if __name__ == "__main__":
    main()
