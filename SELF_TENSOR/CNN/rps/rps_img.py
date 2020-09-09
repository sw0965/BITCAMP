import os
import zipfile

local_zip = 'SELF_TENSOR/rps_data/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('SELF_TENSOR/rps_data/')
zip_ref.close()

local_zip = 'SELF_TENSOR/rps_data/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('SELF_TENSOR/rps_data/')
zip_ref.close()

local_zip = 'SELF_TENSOR/rps_data/rps-validation.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('SELF_TENSOR/rps_data/')
zip_ref.close()