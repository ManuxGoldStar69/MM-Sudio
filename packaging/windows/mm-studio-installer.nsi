; Instalador de Windows para MM Studio (MM Fotografía)
; Compilado con NSIS (makensis) desde macOS o Windows

Unicode True
!include "MUI2.nsh"

!define APP_NAME "MM Studio"
!define APP_PUBLISHER "MM Fotografía"
!define APP_VERSION "0.13.0"
!define APP_EXE "MM Studio.exe"
!define UNINSTALL_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\MMStudio"

Name "${APP_NAME}"
OutFile "..\..\target\windows-installer\MM-Studio-Setup.exe"
InstallDir "$PROGRAMFILES64\${APP_NAME}"
InstallDirRegKey HKLM "Software\MMStudio" "Install_Dir"
RequestExecutionLevel admin
BrandingText "${APP_NAME} — ${APP_PUBLISHER}"

!define MUI_ICON "..\..\assets\app-icon\lightcraft.ico"
!define MUI_UNICON "..\..\assets\app-icon\lightcraft.ico"
!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN "$INSTDIR\${APP_EXE}"
!define MUI_FINISHPAGE_RUN_TEXT "Abrir ${APP_NAME} al finalizar"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "Spanish"
!insertmacro MUI_LANGUAGE "English"

Section "MM Studio (Aplicación Principal)" SecMain
  SectionIn RO
  SetRegView 64
  SetOutPath "$INSTDIR"

  ; Archivo ejecutable principal e icono
  File "/oname=${APP_EXE}" "..\..\target\x86_64-pc-windows-gnu\release\lightcraft.exe"
  File "/oname=mm-studio.ico" "..\..\assets\app-icon\lightcraft.ico"

  ; Catálogos de Filtros Especiales (LUTs 3D), Efectos Mágicos (XMP) y Presets incluidos
  SetOutPath "$INSTDIR\filtros-especiales"
  File /r "..\..\filtros-especiales\*.*"
  SetOutPath "$INSTDIR\efectos-magicos"
  File /r "..\..\efectos-magicos\*.*"
  SetOutPath "$INSTDIR\presets-y-luts"
  File /r "..\..\presets-y-luts\*.*"
  SetOutPath "$INSTDIR"

  ; Guardar ruta de instalación y crear desinstalador
  WriteRegStr HKLM "Software\MMStudio" "Install_Dir" "$INSTDIR"
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  ; Registro en Agregar o quitar programas de Windows
  WriteRegStr HKLM "${UNINSTALL_KEY}" "DisplayName" "${APP_NAME}"
  WriteRegStr HKLM "${UNINSTALL_KEY}" "Publisher" "${APP_PUBLISHER}"
  WriteRegStr HKLM "${UNINSTALL_KEY}" "DisplayVersion" "${APP_VERSION}"
  WriteRegStr HKLM "${UNINSTALL_KEY}" "DisplayIcon" "$INSTDIR\${APP_EXE}"
  WriteRegStr HKLM "${UNINSTALL_KEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegDWORD HKLM "${UNINSTALL_KEY}" "NoModify" 1
  WriteRegDWORD HKLM "${UNINSTALL_KEY}" "NoRepair" 1

  ; Accesos directos en Menú Inicio y Escritorio
  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortcut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}" "" "$INSTDIR\${APP_EXE}" 0
  CreateShortcut "$SMPROGRAMS\${APP_NAME}\Desinstalar ${APP_NAME}.lnk" "$INSTDIR\Uninstall.exe" "" "$INSTDIR\Uninstall.exe" 0
  CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}" "" "$INSTDIR\${APP_EXE}" 0
SectionEnd

Section "Uninstall"
  SetRegView 64
  Delete "$INSTDIR\${APP_EXE}"
  Delete "$INSTDIR\mm-studio.ico"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir /r "$INSTDIR\filtros-especiales"
  RMDir /r "$INSTDIR\efectos-magicos"
  RMDir /r "$INSTDIR\presets-y-luts"
  RMDir "$INSTDIR"

  Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\Desinstalar ${APP_NAME}.lnk"
  RMDir "$SMPROGRAMS\${APP_NAME}"
  Delete "$DESKTOP\${APP_NAME}.lnk"

  DeleteRegKey HKLM "${UNINSTALL_KEY}"
  DeleteRegKey HKLM "Software\MMStudio"
SectionEnd
