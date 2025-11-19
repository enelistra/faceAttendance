from django.db import models
import json

class Employee(models.Model):
    """
    Employee model to store employee information and face encodings
    """
    employee_id = models.CharField(
        max_length=50, 
        unique=True,
        help_text="Unique identifier for the employee"
    )
    
    employee_name = models.CharField(
        max_length=100,
        help_text="Full name of the employee"
    )
    
    face_encoding = models.TextField(
        help_text="Store facial patterns as JSON string"
    )
    
    registration_date = models.DateTimeField(
        auto_now_add=True,
        help_text="Date and time when the employee was registered"
    )
    
    is_active = models.BooleanField(
        default=True,
        help_text="Whether the employee is currently active"
    )
    
    def set_face_encoding(self, encoding):
        """
        Convert facial patterns to JSON string and store in database
        """
        print("💾 Setting face encoding for employee...")
        
        # Convert numpy arrays to lists for JSON serialization
        if 'histogram_features' in encoding:
            encoding['histogram_features'] = [float(x) for x in encoding['histogram_features']]
            print(f"📊 Converted histogram features to list with {len(encoding['histogram_features'])} elements")
        
        if 'lbp_features' in encoding:
            encoding['lbp_features'] = [float(x) for x in encoding['lbp_features']]
            print(f"🔧 Converted LBP features to list with {len(encoding['lbp_features'])} elements")
        
        if 'face_size' in encoding:
            encoding['face_size'] = [int(x) for x in encoding['face_size']]
            print(f"📐 Face size: {encoding['face_size']}")
        
        # Convert to JSON string
        self.face_encoding = json.dumps(encoding)
        print("✅ Face encoding converted to JSON successfully")
    
    def get_face_encoding(self):
        """
        Retrieve and parse facial patterns from JSON string
        """
        if self.face_encoding:
            try:
                encoding = json.loads(self.face_encoding)
                print(f"🔍 Retrieved face encoding for {self.employee_name}")
                return encoding
            except Exception as e:
                print(f"❌ Error parsing face encoding: {e}")
                return None
        else:
            print(f"❌ No face encoding found for {self.employee_name}")
            return None

    def __str__(self):
        """
        String representation of the Employee model
        """
        return f"{self.employee_name} ({self.employee_id}) - {'Active' if self.is_active else 'Inactive'}"

    class Meta:
        """
        Metadata for Employee model
        """
        ordering = ['employee_name']
        verbose_name = 'Employee'
        verbose_name_plural = 'Employees'


class Attendance(models.Model):
    """
    Attendance model to track employee attendance records
    """
    employee = models.ForeignKey(
        Employee,
        on_delete=models.CASCADE,
        help_text="Employee who marked attendance",
        related_name='attendances'
    )
    
    timestamp = models.DateTimeField(
        auto_now_add=True,
        help_text="Date and time when attendance was marked"
    )
    
    status = models.CharField(
        max_length=20,
        default='Present',
        choices=[
            ('Present', 'Present'),
            ('Late', 'Late'),
            ('Absent', 'Absent')
        ],
        help_text="Attendance status"
    )

    def __str__(self):
        """
        String representation of the Attendance model
        """
        return f"{self.employee.employee_name} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {self.status}"

    class Meta:
        """
        Metadata for Attendance model
        """
        ordering = ['-timestamp']
        verbose_name = 'Attendance Record'
        verbose_name_plural = 'Attendance Records'
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['employee', 'timestamp']),
        ]


class ExternalEmployeeMap(models.Model):
    """
    Model to map external employee data to internal face registration
    """
    map_id = models.AutoField(primary_key=True)
    employee_id = models.CharField(max_length=50)
    employee_name = models.CharField(max_length=100)
    is_registered = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.employee_name} ({self.employee_id}) - MapID {self.map_id}"

    class Meta:
        verbose_name = 'External Employee Map'
        verbose_name_plural = 'External Employee Maps'
        ordering = ['-created_at']