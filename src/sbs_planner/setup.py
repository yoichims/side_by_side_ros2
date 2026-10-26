from setuptools import setup

package_name = 'sbs_planner'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Luis Yoichi Morales',
    maintainer_email='yoichims@ieee.org',
    description='Joint planning for a pair of personal mobility vehicles.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'rollout_generator = sbs_planner.rollout_generator_node:main',
            'joint_planner = sbs_planner.joint_planner_node:main',
            'reference_publisher = sbs_planner.reference_publisher_node:main',
            'object_detector = sbs_planner.object_detector_node:main',
            'formation_metrics = sbs_planner.formation_metrics_node:main',
            'velocity_limiter = sbs_planner.velocity_limiter_node:main',
        ],
    },
)
