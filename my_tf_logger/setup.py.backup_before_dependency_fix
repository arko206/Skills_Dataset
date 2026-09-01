from setuptools import setup

package_name = 'my_tf_logger'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
     install_requires=['setuptools', 'numpy', 'zarr',
                      'torch', 'matplotlib', 'gdown', 'diffusers', 'transformers',
                      'collections','json'],
    zip_safe=True,
    maintainer='arka',
    maintainer_email='',
    description='Logs a TF as a 4x4 matrix each timestep.',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'tf_logger = my_tf_logger.tf_logger:main',
            'multi_tf_logger = my_tf_logger.multi_tf_logger:main',
            'base_logger = my_tf_logger.base_logger:main',
            'debug_base_logger = my_tf_logger.debug_base_logger:main',
            'debug_tag_and_base_logger = my_tf_logger.debug_tag_and_base_logger:main',
            'diffusion_agent_dataset = my_tf_logger.diffusion_agent_dataset:main',
            'diffusion_training_go2 = my_tf_logger.diffusion_training_go2:main',
            'eval_diffusion_go2 = my_tf_logger.eval_diffusion_go2:main',
            'rviz_pose_eval_go2 = my_tf_logger.rviz_pose_eval_go2:main',
            'walking_diffus_agent_dataset = my_tf_logger.walking_diffus_agent_dataset:main',
            'walking_diffus_train = my_tf_logger.walking_diffus_train:main',
            'walking_evaluation_param = my_tf_logger.walking_evaluation_param:main',
            'allparams_walking_evaluation_param = my_tf_logger.allparams_walking_evaluation_param:main',
            'stop_base_logger = my_tf_logger.stop_base_logger:main',
            'joints_stop_base_logger = my_tf_logger.joints_stop_base_logger:main',
            'jointsdebug_tag_and_base_logger = my_tf_logger.jointsdebug_tag_and_base_logger:main',
            'demo_base_log = my_tf_logger.demo_base_log:main',
            'reading_base_log_data = my_tf_logger.reading_base_log_data:main',
            'access_deque_data = my_tf_logger.access_deque_data:main',
            'getting_transformations= my_tf_logger.getting_transformations:main',
            'demo_test_log= my_tf_logger.demo_test_log:main'
            



        ],
    },
)

