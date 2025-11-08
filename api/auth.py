# api/auth.py
from fastapi import Header, HTTPException, status, Depends
from firebase_config import firebase_auth # Ambil objek auth yang sudah diinisialisasi

async def verify_token(authorization: str = Header(None)):
    """
    Memverifikasi ID Token dari Header Authorization: Bearer <token>.
    Mengembalikan decoded token jika sukses, atau HTTPException 401 jika gagal.
    """
    if not authorization or "Bearer " not in authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Header Authorization: Bearer <token> missing or invalid."
        )
    
    try:
        # Memisahkan "Bearer" dari token
        token = authorization.split("Bearer ")[1]
        
        if not firebase_auth:
            raise Exception("Firebase Auth is not initialized.")

        # Verifikasi token oleh Firebase
        decoded_token = firebase_auth.verify_id_token(token)
        
        # Kembalikan data yang terdekode (mengandung 'uid', 'email', dll.)
        return decoded_token 
        
    except firebase_auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token tidak valid atau kadaluarsa."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Error Autentikasi: {e}"
        )

# Dependency siap pakai
# Gunakan ini di endpoint yang memerlukan otentikasi: Depends(get_current_user_data)
def get_current_user_data(user_data: dict = Depends(verify_token)):
    return user_data