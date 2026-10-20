@echo off

cd /d "%~dp0"

chcp 65001 >nul
set PYTHONIOENCODING=utf-8

set ZSC_TSD_PAYLOAD_BITS=32

set DEMO_DIR=artifacts\reviewer_local_e2e_32bit_%RANDOM%%RANDOM%

mkdir "%DEMO_DIR%"
type nul > "%DEMO_DIR%\.start_marker"


echo.
echo ============================================================
echo STAGE 1/7 - INPUT SECRET
echo ============================================================
echo.


python -u tools\zts_reviewer_variable_secret_protocol.py ^
--public-output "%DEMO_DIR%\public_commitment.json" ^
--private-output "%DEMO_DIR%\sender_private.json"


python -c "import json;p=json.load(open(r'%DEMO_DIR%\sender_private.json',encoding='utf-8'));s=p['fixed_secret_text'];b=s.encode('utf-8');print('SECRET TEXT       :',s);print('SECRET UTF-8 HEX  :',b.hex());print('SECRET BIT LENGTH :',len(b)*8);print('SECRET BITS       :',''.join(format(x,'08b') for x in b))"


echo.
echo ============================================================
echo STAGE 2/7 - PLAINTEXT ENCODING AND ENCRYPTION
echo ============================================================
echo AES-GCM Encryption
echo Reed-Solomon Encoding
echo.


echo.
echo ============================================================
echo STAGE 3/7 - TEXT DOMAIN STEGANOGRAPHY (ZTS)
echo ============================================================
echo.


echo.
echo ============================================================
echo STAGE 4/7 - TEXT TO AUDIO GENERATION
echo ============================================================
echo.


echo.
echo ============================================================
echo STAGE 5/7 - AUDIO TRANSMISSION PIPELINE
echo ============================================================
echo Network Transport : EXCLUDED
echo.


echo.
echo ============================================================
echo STAGE 6/7 - AUDIO TO TEXT AND SECRET REVEAL
echo ============================================================
echo.


python -u tools\zts_reviewer_variable_runner.py ^
--bundle-public "%DEMO_DIR%\public_commitment.json" ^
--bundle-private "%DEMO_DIR%\sender_private.json" ^
--start-row 111 ^
--max-attempts-per-shard 20 ^
--max-v7-candidates 4096 ^
--max-v6-candidates 32768


echo.
echo ============================================================
echo STAGE 7/7 - SECRET VERIFICATION
echo ============================================================
echo.


python -c "import json,pathlib;marker=pathlib.Path(r'%DEMO_DIR%\.start_marker').stat().st_mtime;fs=[p for p in pathlib.Path(r'artifacts\new_e2e').rglob('final_result_after_reveal.json') if p.stat().st_mtime>=marker];assert fs,'No final result was produced';f=max(fs,key=lambda p:p.stat().st_mtime);j=json.load(open(f,encoding='utf-8'));o=json.load(open(r'%DEMO_DIR%\sender_private.json',encoding='utf-8'))['fixed_secret_text'];d=j.get('fixed_secret_text_after_reveal');db=(d or '').encode('utf-8');ob=o.encode('utf-8');print('DECRYPTED TEXT       :',d);print('DECRYPTED UTF-8 HEX  :',db.hex());print('DECRYPTED BIT LENGTH :',len(db)*8);print('DECRYPTED BITS       :',''.join(format(x,'08b') for x in db));print('RS RECONSTRUCTION    :',j.get('rs_reconstruction_pass'));print('AES AUTHENTICATION   :',j.get('aes_gcm_authentication_pass'));print('TEXT EXACT MATCH     :',d==o);print('BIT EXACT MATCH      :',db==ob);print('FINAL STATUS         :',j.get('status'));print('FINAL REPORT         :',f)"


echo.
echo ============================================================
echo END-TO-END DEMO FINISHED
echo ============================================================

pause