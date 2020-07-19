from django.db import models
from django.contrib.auth.models import User
# Create your models here.
class saved_models(models.Model):
    name=models.CharField(max_length=300)
    user1=models.ForeignKey(User,on_delete=models.CASCADE,blank=True,null=True)
    model=models.FileField(upload_to='saved_models/')
    def __str__(self):
        return self.name
    
    def delete(self, *args, **kwargs):
        self.model.delete()
        super().delete(*args, **kwargs)