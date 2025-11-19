from rest_framework import serializers
from .models import ExternalEmployeeMap

class ExternalEmployeeMapSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExternalEmployeeMap
        fields = ['map_id', 'employee_id', 'employee_name', 'is_registered']