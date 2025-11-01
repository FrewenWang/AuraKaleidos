set(3RDPARTY_XTENSA_LSP_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/xtensa/lsp")

if(AURA_BUILD_XPLORER)
    list(APPEND AURA_LINK_OPTIONS "-lxmem-bank-xtos -lxmem-xtos -lidma-xtos")
    list(APPEND AURA_LINK_OPTIONS "-mlsp=${3RDPARTY_XTENSA_LSP_DIRS}/lsp_xplorer")

    install(DIRECTORY "${3RDPARTY_XTENSA_LSP_DIRS}/lsp_xplorer/" DESTINATION include/3rdparty/lsp)
elseif(AURA_BUILD_XTENSA)
    list(APPEND AURA_LINK_OPTIONS "-Wl,--shared-pagesize=128 -Wl,-pie")
    list(APPEND AURA_LINK_OPTIONS "-mlsp=${3RDPARTY_XTENSA_LSP_DIRS}/lsp_xtensa")

    install(DIRECTORY "${3RDPARTY_XTENSA_LSP_DIRS}/lsp_xtensa/"  DESTINATION include/3rdparty/lsp)
endif()
