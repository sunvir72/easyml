from django.shortcuts import render,HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.contrib import messages

def testapp(request):
    return render(request, 'testapp.html', {})

def log_out(request):
    if not request.user.is_authenticated:
        return render(request, 'testapp.html', {})
    logout(request)
    messages.success(request,('You have successfully logged out'))
    return render(request, 'testapp.html', {})

def Link1(request):
    return render(request, 'Link1.html', {})

def Link11(request):
    if request.method=='POST':
        fname1=request.POST['fname']
        lname1=request.POST['lname']
        email1=request.POST['email']
        psswd1=request.POST['psswd']
        print(lname1)
        if User.objects.filter(username=email1).exists():
            messages.success(request,('Account with this email id already exists!'))
            return render(request, 'Link1.html', {})
        else:
            u=User.objects.create_user(first_name=fname1,last_name=lname1,username=email1,password=psswd1)
            u.save()
            messages.success(request,('You have registered successfully!'))
            return render(request, 'testapp.html', {})
    else:
        return render(request, 'testapp.html', {})

def userlogin(request):
    if request.method=='POST':
        email1=request.POST['email']
        psswd1=request.POST['psswd']
        u = authenticate(username=email1, password=psswd1)
        if u is not None:
            login(request,u)
            messages.success(request,('You have successfully logged in'))
            return render(request, 'testapp.html', {})
        else:
            messages.success(request,('Error! Invalid Email or Password'))
            return render(request, 'Link1-Login.html', {})
    else:
        if request.user.is_authenticated:
            return render(request, 'testapp.html', {})
        return render(request, 'Link1-Login.html', {})
    

