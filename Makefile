run:
	poetry run python -m deep_cluster.main
install:
	poetry install
update:
	poetry update
remove_non_centered_image:
	poetry run python -m deep_cluster.preprocessing.cleanup_dicom